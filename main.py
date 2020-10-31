import model
import arg_parser
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from backbones import CreateModel
import os
import os.path
from logger import Logger
from torch import optim
from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
CS_weights = np.array( (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=np.float32 )
CS_weights = torch.from_numpy(CS_weights)

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

    for epoch in range(args.num_epochs):
        domain_adapter_net.train()
        discriminator_net.train()
        semseg_net.train()
        examples = 0.0
        running_loss = 0.0

        for batch_num in range(args.num_steps):

            start_time = time.time()

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
            loss_G = torch.pow(torch.dist(src_in_trg_labels, src_label_batch), 2.) + torch.pow(1.-discriminator_src_in_trg, 2.)

            loss = loss_D + loss_G

            loss.backward()

            domain_adapter_optimizer.step()
            discriminator_optimizer.step()
            semseg_optimizer.step()

            running_loss += loss.item()
            examples += args.batch_size

            elapsed_time = time.time() - start_time
            logger.info("Elapsed time for batch #%d: %f secs" % (batch_num, elapsed_time))

        logger.info('[%d] train loss: %.15f' %
                     (epoch + 1, running_loss / examples))


if __name__ == "__main__":
    main()
