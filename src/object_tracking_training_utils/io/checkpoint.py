import xml.etree.ElementTree as ET
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best==True, also saves checkpoint + 'best.pth.tar'.

    Args:
        state (dict): contains model's state_dict, may contain other keys such
            as epoch, optimizer state_dict 
        is_best (bool): True if it is the best
            model seen till now 
        checkpoint (string): folder where parameters are
            to be saved.
    Returns:
        None
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}"
              .format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is
    provided, loads state_dict of optimizer assuming it is present in
    checkpoint.
    Args:
        checkpoint (string): filename which needs to be loaded
        model (torch.nn.Module): model for which the parameters are loaded
        optimizer (torch.optim): optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint) if torch.cuda.is_available() \
        else torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint
