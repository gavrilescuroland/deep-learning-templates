import os
from yacs.config import CfgNode as CN

_C = CN()

# System hyperparamters
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_WORKERS = 4

# Data hyperparameters
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 32


# Model hyperparamters
_C.MODEL = CN()
_C.MODEL.NUM_OUTPUTS = 10
_C.MODEL.DROPOUT_RATE = 0.5


# Training hyperparameters
_C.TRAIN = CN()
_C.TRAIN.LEARNING_RATE = 0.001
_C.TRAIN.NUM_EPOCHS = 2
_C.TRAIN.TRAIN_METRICS_FREQUENCY = 200
_C.TRAIN.VALID_METRICS_FREQUENCY = 200
_C.TRAIN.TEST_METRICS_FREQUENCY = 200




cfg = _C