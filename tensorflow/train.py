
import argparse
import logging
import os

import tensorflow as tf
from config.defaults import cfg
from data import dataloaders
from modeling import model
# from engine import trainer
from utils import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/', help="Directory to dataset")
    parser.add_argument('--experiment_dir', default='experiments/base_model', help="Directory to experiment")
    parser.add_argument('--checkpoint_dir', help="Reload weights from .pth.tar file ('best' or 'last')")
    args = parser.parse_args()


    # Set the logger
    utils.set_logger(os.path.join(args.experiment_dir, 'train.log'))


    # Import configs
    config_dir = args.experiment_dir + '/config.yaml'
    cfg.merge_from_file(config_dir)
    cfg.freeze()


    # Fix seed
    tf.random.set_seed(0)


    # Fetch dataloaders
    logging.info("Loading datasets...\n")

    train_dl, test_dl = dataloaders.load_mnist()    


    # Load model, optimizer, loss
    model = model.Net(cfg)

    optimizer = tf.keras.optimizers.Adam(lr=cfg.TRAIN.LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    
    # Train model

    # Keras approach
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'])
    
    model.fit(train_dl, epochs=10)


    # Custom approach
    # trainer.train_and_evaluate(model, train_dl, test_dl, optimizer, loss_fn, args.experiment_dir, args.checkpoint_dir)

