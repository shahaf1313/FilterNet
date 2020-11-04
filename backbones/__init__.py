from backbones.deeplab import Deeplab
from backbones.fcn8s import VGG16_FCN8s
import torch.optim as optim


def CreateModel(args):
    if args.model == 'DeepLab':
        phase = 'test'
        if args.set == 'train' or args.set == 'trainval':
            phase = 'train'
        model = Deeplab(num_classes=args.num_classes, init_weights=args.init_weights, restore_from=args.restore_from, phase=phase)

        if args.set == 'train' or args.set == 'trainval':
            optimizer = optim.SGD(model.optim_parameters(args),
                                  lr=args.generator_lr, momentum=args.momentum, weight_decay=args.weight_decay)
            optimizer.zero_grad()
            return model, optimizer
        else:
            return model
        
    if args.model == 'VGG':
        model = VGG16_FCN8s(num_classes=19, init_weights=args.init_weights, restore_from=args.restore_from)
        if args.set == 'train' or args.set == 'trainval':
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
        else:
            return model


