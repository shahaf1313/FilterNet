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
IMG_PIXELS_NUM = 512*1024*3
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
np.random.RandomState(0)
#todo: add seed to the main (pytorch)


def SaveLossHistory(discriminator_loss_history, generator_loss_history, loss_history_folder, epoch):
    numpy_discriminator_loss = np.array(discriminator_loss_history, dtype=np.float32)
    numpy_generator_loss = np.array(generator_loss_history, dtype=np.float32)
    loss_history_mat = os.path.join(loss_history_folder, 'loss_history_epoch_%d.mat' % epoch)
    sio.savemat(loss_history_mat, {('discriminator_loss_history_epoch_%d' % epoch): numpy_discriminator_loss,
                                   ('generator_loss_history_epoch_%d' % epoch): numpy_generator_loss})

def SetLearnableModels(learnable: list, non_learnable: list):
    for model in non_learnable:
        model.training=False
    for model in learnable:
        model.training = True

def main():
    args = arg_parser.Parse()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    logger = Logger(args.log_dir)
    logger.PrintAndLogArgs(args)
    sourceloader, targetloader = CreateSrcDataLoader(args), CreateTrgDataLoader(args)


    generator = model.DeepLPFNet()
    generator = nn.DataParallel(generator.cuda())
    generator_criterion = model.GeneratorLoss()
    generator_optimizer = optim.Adam(
                                            generator.parameters(),
                                            lr=args.generator_lr,
                                            betas=(0.9, 0.999),
                                            weight_decay=0.0005,
                                            eps=1e-08)
    discriminator = model.Discriminator()
    discriminator = nn.DataParallel(discriminator.cuda())
    discriminator_criterion = model.DiscriminatorLoss()
    discriminator_optimizer = optim.Adam(
                            discriminator.parameters(),
                            lr=args.discriminator_lr,
                            betas=(0.9, 0.999),
                            eps=1e-08)
    semseg_net, semseg_optimizer = CreateModel(args)
    semseg_net = nn.DataParallel(semseg_net.cuda())

    logger.info('######### Network created #########')
    logger.info('Architecture of Generator:\n' + str(generator))
    logger.info('Architecture of Discriminator:\n' + str(discriminator))
    logger.info('Architecture of Backbone net:\n' + str(semseg_net))
    # tb = SummaryWriter(args.tb_logs_dir)
    tb = SummaryWriter()

    for epoch in range(args.num_epochs):
        generator.train()
        discriminator.train()
        semseg_net.train()
        discriminator_loss_history = []
        generator_loss_history = []
        sourceloader_iter, targetloader_iter = iter(sourceloader), iter(targetloader)
        examples, running_loss_discriminator, running_loss_generator, running_time = 0.0, 0.0, 0.0, 0.0
        logger.info('#################[Epoch %d]#################' % (epoch + 1))

        for batch_num in range(args.num_steps):
            start_time = time.time()
            training_discriminator = batch_num % (args.discriminator_iters + args.generator_iters) < args.discriminator_iters
            if training_discriminator:
                SetLearnableModels([discriminator], [semseg_net, generator])
            else:
                SetLearnableModels([semseg_net, generator], [discriminator])
            #todo: what is the 255 ignore label? how should I treat it??
            #todo: should I substract the mean image?? from source? from data??
            src_img, src_lbl, src_shapes, src_names = sourceloader_iter.next()  # new batch source
            trg_img, trg_lbl, trg_shapes, trg_names = targetloader_iter.next()  # new batch target

            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            semseg_optimizer.zero_grad()

            src_input_batch = Variable(src_img, requires_grad=False).cuda()
            src_label_batch = Variable(src_lbl, requires_grad=False).cuda()
            trg_input_batch = Variable(trg_img, requires_grad=False).cuda()
            trg_label_batch = Variable(trg_lbl, requires_grad=False).cuda()

            src_in_trg = generator(src_input_batch)  # G(S,T)
            discriminator_src_in_trg = discriminator(src_in_trg)  # D(G(S,T))
            discriminator_trg = discriminator(trg_input_batch)  # D(T)


            if training_discriminator: #train discriminator
                loss = discriminator_criterion(discriminator_src_in_trg, discriminator_trg)
                discriminator_loss_history.append(loss.item())
                tb.add_scalar('Discriminator Loss Epoch %d' % (epoch+1), loss.item(), len(discriminator_loss_history))
                running_loss_discriminator += loss.item()
            else: #train generator and semseg net
                predicted, loss_seg, loss_ent = semseg_net(src_in_trg, lbl=src_label_batch)  # F(G(S.T))
                # loss = generator_criterion(src_in_trg_labels, src_label_batch, discriminator_trg)
                src_in_trg_labels = torch.argmax(predicted, dim=1)
                loss = generator_criterion(loss_seg, discriminator_trg)
                generator_loss_history.append(loss.item())
                tb.add_scalar('Generator Loss Epoch %d' % (epoch+1), loss.item(), len(generator_loss_history))
                running_loss_generator += loss.item()

            #todo: check with Shady if I need to add the FCN loss also!
            loss.backward()

            if training_discriminator: # train discriminator
                discriminator_optimizer.step()
            else:  # train generator and semseg net
                generator_optimizer.step()
                semseg_optimizer.step()

            running_time += time.time() - start_time
            examples += args.batch_size

            if(batch_num + 1) % args.print_every == 0:
                print_discriminator_loss = running_loss_discriminator/args.print_every
                print_generator_loss = running_loss_generator/args.print_every
                print_time = running_time/args.print_every
                logger.info('[ep %d][it %d][loss discriminator %.8f][loss generator %.8f][time per batch %.2fs]' % \
                      (epoch+1, batch_num+1, print_discriminator_loss, print_generator_loss, print_time))
                running_loss_discriminator, running_loss_generator, running_time= 0.0, 0.0, 0.0
            if(batch_num + 1) % args.save_pred_every == 0:
                pic_file_name = os.path.join(args.pic_dir, 'epoch_%d_batch%d.mat' % (epoch+1, batch_num+1))
                sio.savemat(pic_file_name, {'src_img': src_img[0,:,:,:].cpu().numpy(), 'src_in_trg': src_in_trg[0,:,:,:].detach().cpu().numpy()})

        #todo: Add validation loop!
        logger.info('-----------------------------------Epoch #%d Finished-----------------------------------' % (epoch + 1))
        SaveLossHistory(discriminator_loss_history=discriminator_loss_history,
                        generator_loss_history=generator_loss_history,
                        loss_history_folder=logger.log_folder,
                        epoch=epoch+1)
        tb.flush()

    tb.close()
    logger.info('Finished training.')


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
