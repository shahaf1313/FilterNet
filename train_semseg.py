import arg_parser
from backbones import CreateModel
from constants import NUM_CLASSES, IGNORE_LABEL
from utils import ImageAndLossSaver, compute_cm_batch_torch, compute_iou_torch
from data import CreateSrcDataLoader
import os
from logger import Logger
from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
import time
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# np.random.RandomState(0)
# np.random.seed(0)
# torch.set_deterministic(True)
# torch.backends.cudnn.deterministic = True

def main():
    args = arg_parser.Parse()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    logger = Logger(args.log_dir)
    logger.PrintAndLogArgs(args)
    saver = ImageAndLossSaver(args.tb_logs_dir, logger.log_folder, args.checkpoints_dir, args.save_pics_every)
    source_train_loader = CreateSrcDataLoader(args, 'train_semseg')
    source_val_loader = CreateSrcDataLoader(args, 'val_semseg')
    semseg_net, semseg_optimizer = CreateModel(args)
    semseg_net = nn.DataParallel(semseg_net.cuda())
    semseg_scheduler = torch.optim.lr_scheduler.MultiStepLR(semseg_optimizer, milestones=np.arange(0, args.num_epochs, 10), gamma=0.9)

    logger.info('######### Network created #########')
    logger.info('Architecture of Semantic Segmentation network:\n' + str(semseg_net))

    for epoch in range(args.num_epochs):
        semseg_net.train()
        saver.Reset()
        logger.info('#################[Epoch %d]#################' % (epoch + 1))

        for batch_num, (src_img, src_lbl, _, _) in enumerate(source_train_loader):
            start_time = time.time()
            semseg_optimizer.zero_grad()

            src_input_batch = Variable(src_img, requires_grad=False).cuda()
            src_label_batch = Variable(src_lbl, requires_grad=False).cuda()

            predicted, loss_seg, loss_ent = semseg_net(src_input_batch, lbl=src_label_batch)  # F(G(S.T))
            pred_label = torch.argmax(predicted, dim=1)
            loss = torch.mean(loss_seg + args.entW*loss_ent)

            saver.WriteSemsegLossHistory(args.model, loss.item())
            loss.backward()

            semseg_optimizer.step()
            saver.running_time += time.time() - start_time

            if saver.SaveImagesSemsegIteration:
                saver.SaveTrainSemegImages(epoch,
                                          src_img[0, :, :, :],
                                          src_lbl[0, :, :],
                                          pred_label[0, :, :])

            if (batch_num + 1) % args.print_every == 0:
                logger.info('Finished Batch %d' %(batch_num + 1))

        # Update LR:
        semseg_scheduler.step()

        #Save checkpoint:
        saver.SaveModelsCheckpointSemseg(semseg_net, args.model, epoch)

        #Validation:
        semseg_net.eval()
        rand_samp_inds = np.random.randint(0, len(source_val_loader.dataset), 5)
        rand_batchs = np.floor(rand_samp_inds/args.batch_size).astype(np.int)
        cm = torch.zeros((NUM_CLASSES, NUM_CLASSES)).cuda()
        for val_batch_num, (src_img, src_lbl, _, _) in enumerate(source_val_loader):
            with torch.no_grad():
                src_input_batch = Variable(src_img, requires_grad=False).cuda()
                src_label_batch = Variable(src_lbl, requires_grad=False).cuda()
                pred_softs_batch = semseg_net(src_input_batch)
                pred_batch = torch.argmax(pred_softs_batch, dim=1)
                cm += compute_cm_batch_torch(pred_batch, src_label_batch, IGNORE_LABEL, NUM_CLASSES)
                if (val_batch_num + 1) in rand_batchs:
                    rand_offset = np.random.randint(0, args.batch_size)
                    saver.SaveValidationImages(epoch,
                                               src_input_batch[rand_offset,:,:,:],
                                               src_label_batch[rand_offset,:,:],
                                               pred_batch[rand_offset,:,:])
        iou, miou = compute_iou_torch(cm)
        saver.SaveEpochAccuracy(iou, miou, epoch)
        logger.info('Average accuracy of Epoch #%d on target domain: mIoU = %2f' % (epoch + 1, miou))
        logger.info('-----------------------------------Epoch #%d Finished-----------------------------------' % (epoch + 1))
        del cm, pred_softs_batch, pred_batch

    saver.tb.close()
    logger.info('Finished training.')


if __name__ == "__main__":
    main()

