from nn.ffnn import FFNNet
from nn.dataset import SparseDataset
from nn.execute import Model
from nn.utils import get_dataloader, save_state_dict, load_state_dict

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

import os

def train_nn(feats, labels, net_params, model_out, dir_out):
    '''
    Train neural network classifier
    :param feats: training feats
    :param labels: training labels
    :param net_params: input params for neural net
    :param model_out: fname to save output model
    :param dir_out: directory to save output model
    :return: trained model
    '''

    net = FFNNet(**net_params)

    ds_train = SparseDataset(feats, labels)
    train_loader = get_dataloader(ds_train, shuffle=True)

    model = Model(net)
    net = model.train(train_loader)

    state = {
        'net_params': net_params,
        'state_dict': net.state_dict(),
    }
    save_state_dict(state, fname = model_out, dir_out = dir_out)

    return net

def load_nn(fname, dir_name):
    '''
    Load a trained neural network that has been saved in external file
    :param fname: output model file name
    :param dir_name: directory with mdoel file
    :return: neural network instance of trained model
    '''
    state = load_state_dict(fname, dir_name)

    net_params = state['net_params']

    net = FFNNet(**net_params)
    net.load_state_dict(state['state_dict'])

    return net

def train_baseline(x_train, y_train, model_out, dir_out = '../out/', classifier = 'LR'):
    '''
    Train baseline classifier
    :param x_train: training feats
    :param y_train: training labels
    :param model_out: output file name for saving model
    :param dir_out: model output directory
    :param classifier: NB|LR Naive Bayes or Logistic Regression
    :return: trained classifier
    '''
    if classifier == 'NB':
        model = MultinomialNB(alpha=.01)
    elif classifier == 'LR':
        model = LogisticRegression()

    else:
        raise ValueError("Please input the correct classifier. Currently supported options: (NB|LR)")
    model.fit(x_train, y_train)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    joblib.dump(model, dir_out+model_out)
    return model
