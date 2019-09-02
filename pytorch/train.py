"""Train the model"""

import argparse
import logging
import os

import torch
from torch import nn
import torch.optim as optim
import torchvision

from config.defaults import cfg
from data import dataloaders
from modeling import model
from engine.trainer import train_and_evaluate
from utils import utils


def set_logger(log_path):
    """Set logger to log info in terminal and file `log_path`"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/', help="Directory to dataset")
    parser.add_argument('--experiment_dir', default='experiments/base_model', help="Directory to experiment")
    parser.add_argument('--checkpoint_dir', help="Reload weights from .pth.tar file ('best' or 'last')")
    args = parser.parse_args()


    # Set the logger, import Tensorboard (fixes bugs with the logger)
    set_logger(os.path.join(args.experiment_dir, 'train.log'))
    from torch.utils.tensorboard import SummaryWriter


    # Import configs
    config_dir = args.experiment_dir + '/config.yaml'
    cfg.merge_from_file(config_dir)
    cfg.freeze()


    # Check GPU
    cuda = torch.cuda.is_available()


    # Fix seed
    torch.manual_seed(0)
    if cuda: torch.cuda.manual_seed(0)


    # Fetch dataloaders
    logging.info("Loading datasets...\n")

    dl = dataloaders.fetch_dataloaders(args.data_dir)
    train_dl, valid_dl, test_dl = dl['train'], dl['valid'], dl['test']


    # Define model, optimizer, loss and metrics
    model = model.Net(cfg).cuda() if cuda else model.Net(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()


    # Start Tensorboard, display one batch and the graph
    writer = SummaryWriter()
    images, _ = next(iter(train_dl))
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)


    # Train model
    train_and_evaluate(model, train_dl, valid_dl, test_dl, optimizer, loss_fn, cfg, writer, args.experiment_dir, args.checkpoint_dir)

