"""Train the model"""

import argparse
import logging
import os

import torch
from torch import nn

from config.defaults import cfg
from data import dataloaders
from modeling import model
from engine.inference import evaluate
from utils import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/', help="Directory to dataset")
    parser.add_argument('--experiment_dir', default='experiments/base_model', help="Directory to experiment")
    parser.add_argument('--checkpoint_dir', default='best_model', help="Reload weights from 'best_model' or 'last_model'")
    args = parser.parse_args()


    # Import configs
    config_dir = args.experiment_dir + '/config.yaml'
    cfg.merge_from_file(config_dir)
    cfg.freeze()


    # Check GPU
    cuda = torch.cuda.is_available()



    # Fetch dataloaders
    logging.info("Loading datasets...\n")

    dl = dataloaders.fetch_dataloaders(args.data_dir)
    test_dl = dl['test']


    # Load model, optimizer, loss
    model = model.Net(cfg).cuda() if cuda else model.Net(cfg)
    loss_fn = nn.CrossEntropyLoss()

    # Train model
    evaluate(model, test_dl, loss_fn, args.experiment_dir, args.checkpoint_dir)




