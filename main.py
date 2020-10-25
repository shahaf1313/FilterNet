# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 0.3.1, Python 3.5

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

Instructions:

To get this code working on your system / problem you will need to edit the
data loading functions, as follows:

1. main.py, change the paths for the data directories to point to your data
directory (anything with "/aiml/data")

2. data.py, lines 216, 224, change the folder names of the data x and
output directories to point to your folder names
'''
import model
import arg_parser
import os
import os.path
from logger import Logger
from torch import optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch import nn
import torch
import time
from data import Adobe5kDataLoader, Dataset


def main():
    args = arg_parser.Parse()
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    logger = Logger(args.log_dir)
    logger.PrintAndLogArgs(args)

    training_data_loader = Adobe5kDataLoader(data_dirpath="/home/shahaf/data/GTA5",
                                             img_ids_filepath="/home/shahaf/data_split/GTA5/train.txt")
    training_data_dict = training_data_loader.load_data()
    training_dataset = Dataset(data_dict=training_data_dict, transform=transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
         transforms.ToTensor()]),
                               normaliser=2 ** 8 - 1, is_valid=False)

    training_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True,
                                                       num_workers=4)



    net = model.DeepLPFNet()
    net = nn.DataParallel(net.cuda())

    logger.info('######### Network created #########')
    logger.info('Architecture:\n' + str(net))

    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)

    criterion = None
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    optimizer.zero_grad()
    net.train()

    for epoch in range(num_epoch):

        examples = 0.0
        running_loss = 0.0

        for batch_num, data in enumerate(training_data_loader, 0):

            input_img_batch = Variable(data['input_img'], requires_grad=False).cuda()
            output_img_batch = Variable(data['output_img'], requires_grad=False).cuda()

            start_time = time.time()

            net_output_img_batch = net(input_img_batch)
            net_output_img_batch = torch.clamp(net_output_img_batch, 0.0, 1.0)

            loss = criterion(net_output_img_batch, output_img_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            examples += batch_size

            elapsed_time = time.time() - start_time
            logger.info("Elapsed time for batch #%d: %f secs" % (batch_num, elapsed_time))

        logger.info('[%d] train loss: %.15f' %
                     (epoch + 1, running_loss / examples))


if __name__ == "__main__":
    main()
