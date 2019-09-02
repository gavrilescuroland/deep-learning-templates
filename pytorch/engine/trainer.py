"""Train the model"""

import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from utils import utils


def do_train(model, optimizer, loss_fn, cfg, epoch, writer, train_dl, valid_dl):
    train_loss = valid_loss = 0

    with tqdm(total=len(train_dl)) as t:
        # Train model
        model.train()
        for ii, (train_batch, labels_batch) in enumerate(train_dl):

            if torch.cuda.is_available():
                train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()

            # Forward, loss backprop, update
            output = model(train_batch)
            loss = loss_fn(output, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Incremental running loss
            train_loss += (1 / (ii + 1)) * (loss.data - train_loss)

            # Post metrics on Tensorboard
            if ii % cfg.TRAIN.TRAIN_METRICS_FREQUENCY == 1:
                niter = epoch * len(train_dl) + ii
                writer.add_scalars('Training', {'train_loss': train_loss}, niter)


            t.set_postfix(Loss='{:05.3f}'.format(train_loss))
            t.update()

        # Validate model
        model.eval()
        for ii, (valid_batch, labels_batch) in enumerate(valid_dl):

            if torch.cuda.is_available():
                valid_batch, labels_batch = valid_batch.cuda(), labels_batch.cuda()

            output = model(valid_batch)
            loss = loss_fn(output, labels_batch)

            valid_loss += (1 / (ii + 1)) * (loss.data - valid_loss)

            if ii % cfg.TRAIN.VALID_METRICS_FREQUENCY == 1:
                niter = epoch * len(valid_dl) + ii
                writer.add_scalars('Training', {'valid_loss': valid_loss}, niter)


    return train_loss, valid_loss


def do_evaluate(model, loss_fn, cfg, epoch, writer, test_dl):
    test_loss = correct = total = 0.

    model.eval()
    for ii, (test_batch, labels_batch) in enumerate(test_dl):

        if torch.cuda.is_available():
            test_batch, labels_batch = test_batch.cuda(), labels_batch.cuda()

        output = model(test_batch)
        loss = loss_fn(output, labels_batch)
        test_loss = test_loss + ((1 / (ii + 1)) * (loss.data - test_loss))

        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]

        # Sum correct and total predictions
        correct += np.sum(np.squeeze(pred.eq(labels_batch.data.view_as(pred))).cpu().numpy())
        total += test_batch.size(0)

        if ii % cfg.TRAIN.TEST_METRICS_FREQUENCY == 1:
            niter = epoch * len(test_dl) + ii
            writer.add_scalar('Evaluation/test_loss', test_loss, niter)
            writer.add_scalar('Evaluation/accuracy', 100.*correct/total, niter)


    return test_loss, 100.*correct/total



def train_and_evaluate(model, train_dl, valid_dl, test_dl, optimizer, loss_fn, cfg, writer, experiment_dir, checkpoint_model=None):

    # If checkpoint given, reload parameters (and optimizer)
    if checkpoint_model is not None:
        checkpoint_path = os.path.join(experiment_dir, checkpoint_model + '.pth.tar')
        logging.info("Restoring parameters from {}".format(checkpoint_path))
        utils.load_checkpoint(checkpoint_path, model, optimizer)

    valid_loss_min = np.Inf

    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        logging.info("Epoch {}/{}".format(epoch+1, cfg.TRAIN.NUM_EPOCHS))

        train_loss, valid_loss = do_train(model, optimizer, loss_fn, cfg, epoch, writer, train_dl, valid_dl)

        test_loss, accuracy = do_evaluate(model, loss_fn, cfg, epoch, writer, test_dl)


        # Log epoch metrics
        logging.info('Train loss: {}\nValid loss: {}\nTest loss:  {}\nAccuracy: {}%\n'.format(
            round(train_loss.item(),3), round(valid_loss.item(),3), round(test_loss.item(),3), accuracy))



        # Save parameters
        is_best = valid_loss <= valid_loss_min
        utils.save_checkpoint(state={'epoch': epoch + 1,
                                     'state_dict': model.state_dict(),
                                     'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              experiment_dir=experiment_dir)








