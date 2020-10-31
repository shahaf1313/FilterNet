import argparse


def Parse():
    parser = argparse.ArgumentParser(description="Train the DeepLPF neural network on image pairs")
    #Basics:
    parser.add_argument("--num_epochs", type=int, required=False, help="Number of epochs (default 100000)", default=100000)
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size on training (default 1)", default=1)
    parser.add_argument("--num_steps", type=int, default=150000, help="Number of training steps per epoch (default 150000)")
    parser.add_argument("--log_dir", type=str, required=False, help="Path to logs directory", default=None)
    parser.add_argument("--gpus", type=str, required=False, help="String that contains available GPUs to use", default="0")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of threads for each worker")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Initial learning rate for the segmentation network")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
    parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate (only for deeplab).")

    #Data and sanpshots:
    parser.add_argument("--model", type=str, required=False, default='DeepLab', help="available options : DeepLab and VGG")
    parser.add_argument("--source", type=str, required=False, default='gta5', help="source dataset : gta5 or synthia")
    parser.add_argument("--target", type=str, required=False, default='cityscapes', help="target dataset : cityscapes")
    parser.add_argument("--snapshot_dir", type=str, required=False, default='../checkpoints/FDA', help="Where to save snapshots of the model.")
    parser.add_argument("--data_dir", type=str, required=False, default='../data_semseg/GTA5', help="Path to the directory containing the source dataset.")
    parser.add_argument("--data_list", type=str, required=False, default='./dataset/gta5_list/train.txt', help="Path to the listing of images in the source dataset.")
    parser.add_argument("--data_dir_target", required=False, type=str, default='../data_semseg/cityscapes', help="Path to the directory containing the target dataset.")
    parser.add_argument("--data_list_target", required=False, type=str, default='./dataset/cityscapes_list/train.txt', help="list of images in the target dataset.")
    parser.add_argument("--set", type=str, default='train', help="Choose adaptation set: train, trainval, eval")

    #Save and Restore:
    parser.add_argument("--num_classes", type=int, required=False, default=19, help="Number of classes for cityscapes.")
    parser.add_argument("--init_weights", type=str, required=False, default=None, help="initial model.")
    parser.add_argument("--restore_from", type=str, required=False, default=None, help="Where restore model parameters from.")
    parser.add_argument("--save_pred_every", type=int, required=False, default=2500, help="Save summaries and checkpoint every defined steps number.")
    parser.add_argument("--print_every", type=int, required=False, default=100, help="Print loss data frequency")
    parser.add_argument("--matname", type=str, required=False, default='loss_log.mat', help=".mat file name to save loss")
    parser.add_argument("--pic_dir", type=str, required=False, default=None, help="dir to save sample images (source, target, etc..)")

    return parser.parse_args()

