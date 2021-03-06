import argparse


def Parse():
    parser = argparse.ArgumentParser(description="Train the DeepLPF neural network on image pairs")
    #Basics:
    parser.add_argument("--num_epochs", type=int, required=False, help="Number of epochs (default 100000)", default=100000)
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size on training (default 1)", default=1)
    parser.add_argument("--log_dir", type=str, required=False, help="Path to logs directory", default=None)
    parser.add_argument("--gpus", type=str, required=False, help="String that contains available GPUs to use", default="0")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of threads for each worker")
    parser.add_argument("--generator_lr", type=float, default=2.5e-4, help="Initial learning rate for the segmentation network")
    parser.add_argument("--discriminator_lr", type=float, default=1e-5, help="Initial learning rate for the discriminator network")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
    parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate (only for deeplab).")
    parser.add_argument("--entW", type=float, default=0.007, help="weight for entropy")
    parser.add_argument("--ita", type=float, default=2.0, help="ita for robust entropy")
    parser.add_argument("--discriminator_iters", type=int, default=10, help="Number of iterations to teach discriminator before switching to generator.")
    parser.add_argument("--generator_iters", type=int, default=300, help="Number of iterations to teach generator before switching to discriminator.")
    parser.add_argument("--generator_boost", type=int, default=200, help="Number of iterations to BOOST generator before training to discriminator.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")

    #Data and sanpshots:
    parser.add_argument("--model", type=str, required=False, default='DeepLab', help="available options : DeepLab and VGG")
    parser.add_argument("--source", type=str, required=False, default='gta5', help="source dataset : gta5 or synthia")
    parser.add_argument("--target", type=str, required=False, default='cityscapes', help="target dataset : cityscapes")
    parser.add_argument("--checkpoints_dir", type=str, required=False, default='../checkpoints/FDA', help="Where to save snapshots of the model.")
    parser.add_argument("--tb_logs_dir", type=str, required=False, default='./runs', help="Path to Tensorboard logs dir.")
    parser.add_argument("--data_dir", type=str, required=False, default='../data_semseg/GTA5', help="Path to the directory containing the source dataset.")
    parser.add_argument("--data_list", type=str, required=False, default='./dataset/gta5_list/train.txt', help="Path to the listing of images in the source dataset.")
    parser.add_argument("--data_dir_target", required=False, type=str, default='../data_semseg/cityscapes', help="Path to the directory containing the target dataset.")
    parser.add_argument("--data_list_target_train", required=False, type=str, default='./dataset/cityscapes_list/train.txt', help="list of images in the target dataset.")
    parser.add_argument("--data_list_target_val", required=False, type=str, default='./dataset/cityscapes_list/val.txt', help="list of images in the target dataset.")

    #Save and Restore:
    parser.add_argument("--save_pics_every", type=int, required=False, default=2500, help="Save pictures of source and target.")
    parser.add_argument("--save_checkpoint", type=int, required=False, default=2500, help="Save summaries and checkpoint every defined steps number.")
    parser.add_argument("--print_every", type=int, required=False, default=100, help="Print loss data frequency")

    return parser.parse_args()

