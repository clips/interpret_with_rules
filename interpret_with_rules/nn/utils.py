import os
import multiprocessing

from torch.utils.data import DataLoader
import torch

def get_dataloader(dataset, shuffle, batch_size = 16, n_workers = multiprocessing.cpu_count()):
    return DataLoader(dataset = dataset, shuffle = shuffle, batch_size = batch_size, num_workers = n_workers)

def save_model(net, fname, dir_out ='../out/'):
    '''
    Save parameters of a trained pytorch model
    :param net: Trained Pytorch model
    :param fname: Output file name
    :param dir_out: Output directory
    '''
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    torch.save(net.state_dict(), dir_out + fname)

def load_model(net, fname, dir_out = '../out/'):
    '''
    Load pytorch model parameters into a preinitialized model
    :param net: Preinitialized network
    :param fname: Save parameter file name
    :param dir_out: File directory
    :return: model with the loaded parameters
    '''
    if not os.path.exists(dir_out+fname):
        raise FileNotFoundError("Please enter the correct file path")

    net.load_state_dict(torch.load(dir_out+fname))
    return net

def save_state_dict(state, fname, dir_out = '../out/'):
    '''
    Save model state along with relevant architecture parameters as a state dictionary
    :param state: state dictionary with relevant details (e.g. network arch, epoch, model states and optimizer states)
    :param fname: out file name
    :param dir_out: out directory
    '''
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    torch.save(state, dir_out + fname)

def load_state_dict(fname, dir_in ='../out/'):
    '''
    Load dictionary of model state and arch params
    :param fname: state file name to load
    :param dir_in: directory with filename
    '''
    if not os.path.exists(dir_in + fname):
        raise FileNotFoundError("Please enter the correct file path")

    return torch.load(dir_in + fname)



