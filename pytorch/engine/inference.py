"""Train the model"""

import logging
import os

import numpy as np
import torch
import time

from utils import utils



def evaluate(model, test_dl, loss_fn, experiment_dir, checkpoint_model=None):

    # If checkpoint given, reload parameters
    if checkpoint_model is None:
        print('Error: Model to be evaluated has not been loaded.')

    else:
        checkpoint_path = os.path.join(experiment_dir, checkpoint_model + '.pth.tar')
        logging.info("Restoring parameters from {}".format(checkpoint_path))
        utils.load_checkpoint(checkpoint_path, model)

        correct = total = 0.

        model.eval()

        test_batch, labels_batch = next(iter(test_dl))
        if torch.cuda.is_available():
            test_batch, labels_batch = test_batch.cuda(), labels_batch.cuda()

        start_time = time.time()
        output = model(test_batch)
        end_time = time.time()
        test_loss = loss_fn(output, labels_batch).data

        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]

        # Sum correct and total predictions
        correct += np.sum(np.squeeze(pred.eq(labels_batch.data.view_as(pred))).cpu().numpy())
        total += test_batch.size(0)


        # Print metrics
        print("Metrics for one batch:")
        print("Test Loss: ", round(test_loss.item(),3))
        print("Accuracy: ",round(100.*correct/total, 3))
        print("Inference time: {}s ".format(round(end_time-start_time, 3)))







