import model
import arg_parser
from utils import *
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
from torch.utils.tensorboard import SummaryWriter

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_PIXELS_NUM = 512*1024*3
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark =False
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.deterministic = True
# np.random.RandomState(0)
# np.random.seed(0)

def main():
    args = arg_parser.Parse()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    logger = Logger(args.log_dir)
    logger.PrintAndLogArgs(args)
    saver = ImageAndLossSaver(SummaryWriter(), logger.log_folder, args.save_pics_every)

    sourceloader, targetloader = CreateSrcDataLoader(args), CreateTrgDataLoader(args)


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
        sourceloader_iter, targetloader_iter = iter(sourceloader), iter(targetloader)
        logger.info('#################[Epoch %d]#################' % (epoch + 1))

        for batch_num in range(args.num_steps):
            start_time = time.time()
            training_discriminator = (batch_num >= args.generator_boost) and (batch_num-args.generator_boost) % (args.discriminator_iters + args.generator_iters) < args.discriminator_iters
            #todo: should I substract the mean image?? from source? from data?? -> try and find out!
            src_img, src_lbl, src_shapes, src_names = sourceloader_iter.next()  # new batch source
            trg_img, trg_lbl, trg_shapes, trg_names = targetloader_iter.next()  # new batch target

            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            semseg_optimizer.zero_grad()

            src_input_batch = Variable(src_img, requires_grad=False).cuda()
            src_label_batch = Variable(src_lbl, requires_grad=False).cuda()
            trg_input_batch = Variable(trg_img, requires_grad=False).cuda()
            # trg_label_batch = Variable(trg_lbl, requires_grad=False).cuda()

            src_in_trg = generator(src_input_batch, trg_input_batch)  # G(S,T)
            discriminator_src_in_trg = discriminator(src_in_trg)  # D(G(S,T))
            discriminator_trg = discriminator(trg_input_batch)  # D(T)


            if training_discriminator: #train discriminator
                loss = discriminator_criterion(discriminator_src_in_trg, discriminator_trg)
            else: #train generator and semseg net
                #todo: check if losses are conncted to the computational graph! seems they don't...
                predicted, loss_seg, loss_ent = semseg_net(src_in_trg, lbl=src_label_batch)  # F(G(S.T))
                src_in_trg_labels = torch.argmax(predicted, dim=1)
                loss = generator_criterion(loss_seg, loss_ent, args.entW, discriminator_trg)

            saver.WriteLossHistory(training_discriminator, epoch, loss.item())
            loss.backward()

            if training_discriminator: # train discriminator
                discriminator_optimizer.step()
            else:  # train generator and semseg net
                generator_optimizer.step()
                semseg_optimizer.step()

            saver.running_time += time.time() - start_time

            if (not training_discriminator) and saver.SaveIteration:
                saver.SaveImages(epoch, batch_num,
                             src_img[0, :, :, :],
                             src_in_trg[0, :, :, :],
                             src_lbl[0, :, :],
                             src_in_trg_labels[0, :, :])

            if (batch_num + 1) % args.print_every == 0:
                logger.PrintAndLogData(saver, epoch, batch_num, args.print_every)

            if (batch_num + 1) % args.save_checkpoint == 0:
                #todo: add save checkpoint functionallry to saver!
                #todo: saver.SaveCheckpoint(epoch)
                pass

        #todo: Add validation loop!
        logger.info('-----------------------------------Epoch #%d Finished-----------------------------------' % (epoch + 1))
        saver.SaveLossHistory(logger.log_folder, epoch)

    saver.tb.close()
    logger.info('Finished training.')


if __name__ == "__main__":
    main()
