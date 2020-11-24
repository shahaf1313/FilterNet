from backbones.deeplab import Deeplab
from backbones.fcn8s import VGG16_FCN8s
import torch.optim as optim
from constants import NUM_CLASSES


def CreateModel(args):
    model, optimizer = None, None
    if args.model == 'DeepLab':
        model = Deeplab(num_classes=NUM_CLASSES)
        optimizer = optim.SGD(model.optim_parameters(args),
                              lr=args.generator_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer.zero_grad()

    if args.model == 'VGG':
        model = VGG16_FCN8s(num_classes=NUM_CLASSES)
        optimizer = optim.Adam(
        [
            {'params': model.get_parameters(bias=False)},
            {'params': model.get_parameters(bias=True),
             'lr': args.generator_lr * 2}
        ],
        lr=args.generator_lr,
        betas=(0.9, 0.99))
        optimizer.zero_grad()

    return model, optimizer


