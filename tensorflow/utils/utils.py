import logging
import os
import shutil



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


def save_checkpoint(state, is_best, experiment_dir):
    """Save checkpoint model. Additionally, save it as 'best_model' if so"""
    filepath = os.path.join(experiment_dir, 'last_model.pth.tar')
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(experiment_dir, 'best_model.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters from file_path (Optional: restore optimizer)"""

    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint