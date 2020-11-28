import model
import arg_parser
from constants import NUM_CLASSES, IGNORE_LABEL
from utils import ImageAndLossSaver, compute_cm_batch_torch, compute_iou_torch
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from backbones import CreateModel
import os
from logger import Logger
from torch import optim
from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
import time
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.RandomState(0)
np.random.seed(0)
torch.set_deterministic(True)
torch.backends.cudnn.deterministic = True


def main():
    args = arg_parser.Parse()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    logger = Logger(args.log_dir)
    logger.PrintAndLogArgs(args)
    saver = ImageAndLossSaver(args.tb_logs_dir, logger.log_folder, args.checkpoints_dir, args.save_pics_every)
    source_loader, target_train_loader, target_eval_loader = CreateSrcDataLoader(args), CreateTrgDataLoader(args, 'train'), CreateTrgDataLoader(args, 'val')
    epoch_size = np.maximum(len(target_train_loader.dataset), len(source_loader.dataset))
    steps_per_epoch = int(np.floor(epoch_size / args.batch_size))
    source_loader.dataset.SetEpochSize(epoch_size)
    target_train_loader.dataset.SetEpochSize(epoch_size)

    generator = model.DeepLPFNet()
    generator = nn.DataParallel(generator.cuda())
    generator_criterion = model.GeneratorLoss()
    generator_optimizer = optim.Adam(   generator.parameters(),
                                        lr=args.generator_lr,
                                        betas=(0.9, 0.999),
                                        eps=1e-08)
    discriminator = model.Discriminator()
    discriminator = nn.DataParallel(discriminator.cuda())
    discriminator_criterion = model.DiscriminatorLoss()
    discriminator_optimizer = optim.Adam(   discriminator.parameters(),
                                            lr=args.discriminator_lr,
                                            betas=(0.9, 0.999),
                                            eps=1e-08)
    semseg_net, semseg_optimizer = CreateModel(args)
    semseg_net = nn.DataParallel(semseg_net.cuda())

    logger.info('######### Network created #########')
    logger.info('Architecture of Generator:\n' + str(generator))
    logger.info('Architecture of Discriminator:\n' + str(discriminator))
    logger.info('Architecture of Backbone net:\n' + str(semseg_net))

    for epoch in range(args.num_epochs):
        generator.train()
        discriminator.train()
        semseg_net.train()
        saver.Reset()
        discriminate_src = True
        source_loader_iter, target_train_loader_iter, target_eval_loader_iter = iter(source_loader), iter(target_train_loader), iter(target_eval_loader)
        logger.info('#################[Epoch %d]#################' % (epoch + 1))

        for batch_num in range(steps_per_epoch):
            start_time = time.time()
            training_discriminator = (batch_num >= args.generator_boost) and (batch_num-args.generator_boost) % (args.discriminator_iters + args.generator_iters) < args.discriminator_iters
            src_img, src_lbl, src_shapes, src_names = source_loader_iter.next()  # new batch source
            trg_eval_img, trg_eval_lbl, trg_shapes, trg_names = target_train_loader_iter.next()  # new batch target

            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            semseg_optimizer.zero_grad()

            src_input_batch = Variable(src_img, requires_grad=False).cuda()
            src_label_batch = Variable(src_lbl, requires_grad=False).cuda()
            trg_input_batch = Variable(trg_eval_img, requires_grad=False).cuda()
            # trg_label_batch = Variable(trg_lbl, requires_grad=False).cuda()
            src_in_trg = generator(src_input_batch, trg_input_batch)  # G(S,T)

            if training_discriminator: #train discriminator
                if discriminate_src == True:
                    discriminator_src_in_trg = discriminator(src_in_trg)  # D(G(S,T))
                    discriminator_trg = None  # D(T)
                else:
                    discriminator_src_in_trg = None  # D(G(S,T))
                    discriminator_trg = discriminator(trg_input_batch)  # D(T)
                discriminate_src = not discriminate_src
                loss = discriminator_criterion(discriminator_src_in_trg, discriminator_trg)
            else: #train generator and semseg net
                discriminator_trg = discriminator(trg_input_batch)  # D(T)
                predicted, loss_seg, loss_ent = semseg_net(src_in_trg, lbl=src_label_batch)  # F(G(S.T))
                src_in_trg_labels = torch.argmax(predicted, dim=1)
                loss = generator_criterion(loss_seg, loss_ent, args.entW, discriminator_trg)

            saver.WriteLossHistory(training_discriminator, loss.item())
            loss.backward()

            if training_discriminator: # train discriminator
                discriminator_optimizer.step()
            else:  # train generator and semseg net
                generator_optimizer.step()
                semseg_optimizer.step()

            saver.running_time += time.time() - start_time

            if (not training_discriminator) and saver.SaveImagesIteration:
                saver.SaveTrainImages(epoch,
                                      src_img[0, :, :, :],
                                      src_in_trg[0, :, :, :],
                                      src_lbl[0, :, :],
                                      src_in_trg_labels[0, :, :])

            if (batch_num + 1) % args.print_every == 0:
                logger.PrintAndLogData(saver, epoch, batch_num, args.print_every)

            if (batch_num + 1) % args.save_checkpoint == 0:
                saver.SaveModelsCheckpoint(semseg_net, discriminator, generator, epoch, batch_num)

        #Validation:
        semseg_net.eval()
        rand_samp_inds = np.random.randint(0, len(target_eval_loader.dataset), 5)
        rand_batchs = np.floor(rand_samp_inds/args.batch_size).astype(np.int)
        cm = torch.zeros((NUM_CLASSES, NUM_CLASSES)).cuda()
        for val_batch_num, (trg_eval_img, trg_eval_lbl, _, _) in enumerate(target_eval_loader):
            with torch.no_grad():
                trg_input_batch = Variable(trg_eval_img, requires_grad=False).cuda()
                trg_label_batch = Variable(trg_eval_lbl, requires_grad=False).cuda()
                pred_softs_batch = semseg_net(trg_input_batch)
                pred_batch = torch.argmax(pred_softs_batch, dim=1)
                cm += compute_cm_batch_torch(pred_batch, trg_label_batch, IGNORE_LABEL, NUM_CLASSES)
                print('Validation: saw', val_batch_num*args.batch_size, 'examples')
                if (val_batch_num + 1) in rand_batchs:
                    rand_offset = np.random.randint(0, args.batch_size)
                    saver.SaveValidationImages(epoch,
                                               trg_input_batch[rand_offset,:,:,:],
                                               trg_label_batch[rand_offset,:,:],
                                               pred_batch[rand_offset,:,:])
        iou, miou = compute_iou_torch(cm)
        saver.SaveEpochAccuracy(iou, miou, epoch)
        logger.info('Average accuracy of Epoch #%d on target domain: mIoU = %2f' % (epoch + 1, miou))
        logger.info('-----------------------------------Epoch #%d Finished-----------------------------------' % (epoch + 1))
        del cm, trg_input_batch, trg_label_batch, pred_softs_batch, pred_batch

    saver.tb.close()
    logger.info('Finished training.')


if __name__ == "__main__":
    main()
