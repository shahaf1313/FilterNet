import argparse


def Parse():
    parser = argparse.ArgumentParser(description="Train the DeepLPF neural network on image pairs")
    parser.add_argument("--num_epoch", type=int, required=False, help="Number of epochs (default 100000)", default=100000)
    parser.add_argument("--log_dir", type=str, required=False, help="Path to logs directory", default=None)
    parser.add_argument("--gpus", type=str, required=False, help="String that contains available GPUs to use.", default="0")
    parser.add_argument("--batch_size",  type=int, required=False, help="Batch size on training.", default=1)
    return parser.parse_args()
