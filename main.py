import model
import arg_parser
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from backbones import CreateModel
import os
from PIL import Image
import os.path
from logger import Logger
from torch import optim
from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
import time
import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
CS_weights = np.array( (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=np.float32 )
CS_weights = torch.from_numpy(CS_weights)


def SaveLossHistory(loss_history, loss_history_folder, epoch):
    numpy_loss_history = np.array(loss_history, dtype=np.float32)
    loss_history_mat = os.path.join(loss_history_folder, 'loss_history_epoch_%d.mat' % epoch)
    sio.savemat(loss_history_mat, {('loss_history_epoch_%d' % epoch): numpy_loss_history})


def main():
    args = arg_parser.Parse()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    logger = Logger(args.log_dir)
    logger.PrintAndLogArgs(args)

    sourceloader, targetloader = CreateSrcDataLoader(args), CreateTrgDataLoader(args)
    sourceloader_iter, targetloader_iter = iter(sourceloader), iter(targetloader)


    domain_adapter_net = model.DeepLPFNet()
    domain_adapter_net = nn.DataParallel(domain_adapter_net.cuda())
    domain_adapter_optimizer = optim.Adam(
                                            filter(lambda p: p.requires_grad, domain_adapter_net.parameters()),
                                            lr=args.learning_rate,
                                            betas=(0.9, 0.999),
                                            eps=1e-08)
    discriminator_net = model.Discriminator()
    discriminator_net = nn.DataParallel(discriminator_net.cuda())
    discriminator_optimizer = optim.Adam(
                            filter(lambda p: p.requires_grad, discriminator_net.parameters()),
                            lr=args.learning_rate,
                            betas=(0.9, 0.999),
                            eps=1e-08)
    semseg_net, semseg_optimizer = CreateModel(args)
    semseg_net = nn.DataParallel(semseg_net.cuda())

    logger.info('######### Network created #########')
    logger.info('Architecture of Domain Adapter:\n' + str(domain_adapter_net))
    logger.info('Architecture of Discriminator:\n' + str(discriminator_net))
    logger.info('Architecture of backbone net:\n' + str(semseg_net))

    # tb = SummaryWriter()
    # tb.add_scalar("loss", loss)
    loss_history = []
    eps = 1e-4
    for epoch in range(args.num_epochs):
        domain_adapter_net.train()
        discriminator_net.train()
        semseg_net.train()
        examples = 0.0
        running_loss = 0.0
        running_time = 0.0

        for batch_num in range(args.num_steps):

            start_time = time.time()
            #todo: what is the 255 ignore label? how should I treat it??
            #todo: should I substract the mean image?? from source? from data??
            src_img, src_lbl, src_shapes, src_names = sourceloader_iter.next()  # new batch source
            tag_img, trg_lbl, trg_shapes, trg_names = targetloader_iter.next()  # new batch target

            domain_adapter_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            semseg_optimizer.zero_grad()

            src_input_batch = Variable(src_img, requires_grad=False).cuda()
            src_label_batch = Variable(src_lbl, requires_grad=False).cuda()
            trg_input_batch =  Variable(tag_img, requires_grad=False).cuda()

            src_in_trg = domain_adapter_net(src_input_batch) #G(S,T)
            discriminator_src_in_trg = discriminator_net(src_in_trg) #D(G(S,T))
            discriminator_trg = discriminator_net(trg_input_batch) #D(T)
            loss_D = torch.pow(discriminator_src_in_trg, 2.) + torch.pow(1.-discriminator_trg, 2.)

            src_in_trg_labels = semseg_net(src_in_trg, lbl=src_label_batch) #F(G(S.T))
            src_in_trg_labels = torch.argmax(src_in_trg_labels, dim=1)
            loss_G = torch.pow(torch.dist(src_in_trg_labels, src_label_batch), 2.) + torch.pow(1.-discriminator_src_in_trg, 2.)

            #todo: check with Shady if I need to add the FCN loss also!
            loss = eps * (loss_D + loss_G) #+ semseg_net.FCN8s.loss_seg
            loss = torch.mean(loss)
            loss.backward()

            domain_adapter_optimizer.step()
            discriminator_optimizer.step()
            semseg_optimizer.step()

            elapsed_time = time.time() - start_time
            running_time += elapsed_time
            running_loss += loss.item()
            examples += args.batch_size

            if(batch_num + 1) % args.print_every == 0:
                logger.info('[it %d][loss %.4f][time per batch %.2fs]' % \
                      (batch_num+1, running_loss/args.print_every, running_time/(batch_num+1.)))
                loss_history.append(running_loss/args.print_every)
                running_time = 0.0
                running_loss = 0.0
            if(batch_num + 1) % args.save_pred_every == 0:
                pic_file_name = os.path.join(args.pic_dir, 'epoch_%d_batch%d.mat' % (epoch+1, batch_num+1))
                sio.savemat(pic_file_name, {'src_img': src_img[0,:,:,:].cpu().numpy(), 'src_in_trg': src_in_trg[0,:,:,:].detach().cpu().numpy()})

        #todo: Add validation loop!
        logger.info('-----------------------------------Epoch #%d Finished-----------------------------------' % (epoch + 1))
        SaveLossHistory(loss_history, logger.log_folder, epoch+1)


# color coding of semantic classes
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


if __name__ == "__main__":
    main()
