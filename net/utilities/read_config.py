import pandas as pd
import numpy as np
import yaml

# Config file has: data directories, architecture of the network and hyperparameters for the training session.
def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        data_dir = config['DATA']['DATA_DIR']
        mode = config['DATA']['MODE']
        arch = config['MODEL']['ARCH']
        batch_size = config['TRAIN']['BATCH_SIZE']
        epochs = config['TRAIN']['EPOCHS']
        learning_rate = config['TRAIN']['LR']
    return {'data_dir': data_dir, 'mode': mode, 'arch': arch, 'batch_size': batch_size, 'epochs': epochs, 'lr': learning_rate}