import os
import pickle
import gzip

def write_data(data, fname, dir_name = '../out/'):
    """
    Write data as gzip pickle object
    :param data: data to write
    :param fname: output file name
    :param dir_name: output directory
    """

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with gzip.GzipFile(dir_name+fname, 'w') as fout:
        pickle.dump(data, fout)

def load_data(fname, dir_name = '../out/'):
    """
    Load pickled gzip file
    :param fname: file name to load object from
    :param dir_name: directory name
    :return: loaded object
    """

    with gzip.GzipFile(dir_name+fname, 'w') as fout:
        return pickle.load(fout)