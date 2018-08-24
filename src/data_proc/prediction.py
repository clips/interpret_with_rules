from nn.dataset import SparseDataset
from nn.utils import get_dataloader
from nn.execute import Model

def pred_nn(net, x_test):
    '''
    Get predicted classes and probability of prediction given a Pytorch model and test features
    :param net: Trained Pytorch model
    :param x_test: test feature matrix
    :return: Arrays of predicted classes, and prediction probabilities of all classes for all instances
    '''

    ds = SparseDataset(x_test)
    dataloader = get_dataloader(ds, shuffle=False)

    model = Model(net)
    return model.predict(dataloader)

def pred_prob_nn(net, x_test):
    '''
    :param net: trained Pytorch model
    :param x_test: test feature matrix
    :return: Prediction probability of all output classes for all instances (n_inst, n_classes)
    '''
    ds = SparseDataset(x_test)
    dataloader = get_dataloader(ds, shuffle=False)

    model = Model(net)
    return model.predict_prob(dataloader)

def pred_baseline(model, x_test):
    '''
    Return predictions of the baseline model
    :param model: Trained sklearn baseline (LR|NB) model
    :param x_test: test features
    :return: array of the predicted classes of every instance
    '''

    return model.predict(x_test)