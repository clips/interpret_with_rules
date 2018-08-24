from torch.autograd import Variable
import torch

import numpy as np


class Model(object):
    '''
    Pytorch model class
    '''
    def __init__(self, net):
        self.net = net

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.net = self.net.to(self.device)

    def train(self,
            trainloader,
            int_act_fn = 'relu',
            output_act_fn='softmax',
            loss_name = 'CrossEntropyLoss',
            optimizer_name ='Adam',
            n_epochs = 50):
        '''
        Trains a pytorch model
        :param trainloader: dataloader that yields input features and labels
        :param int_act_fn: activation function for hidden layers
        :param output_act_fn: activation function for output layer
        :param loss_name: Name of the loss function to use
        :param optimizer_name: Name of the optimizer to use
        :param n_epochs: number of epochs to train a model for
        :param cuda: True if cuda is enabled on the device
        :return: trained model
        '''

        for epoch in range(n_epochs):
            running_loss = 0.0

            for idx, (inputs, labels) in enumerate(trainloader):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs.requires_grad_()

                fwd_out = self.net.forward(inputs, int_act_fn, output_act_fn)
                loss = self.net.backprop(fwd_out, labels, loss_name, optimizer_name)

                # print statistics
                running_loss += loss.item()

                if idx % 100 == 99:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, idx + 1, running_loss / 2000))
                    running_loss = 0.0

        return self.net


    def predict(self, testloader, int_act_fn = 'relu', out_act_fn='softmax'):
        '''
        Predict output of test data
        :param testloader: dataloader which yields test features
        :param int_act_fn: activation function for hidden layers
        :param out_act_fn: activation function for output layers
        :return: array of predicted labels, array with prob of every class for every instance
        '''

        out_probs = list()
        out_labels = list()

        for inputs in testloader:

            inputs = inputs.to(self.device)

            out_data, pred = self._get_prediction(inputs, int_act_fn, out_act_fn)

            for i, row in enumerate(out_data):
                out_probs.append(row)
                out_labels.append(pred[i])

        return np.array(out_labels), np.vstack(out_probs)

    def _get_prediction(self, inputs, int_act_fn, out_act_fn):

        fwd_outputs = self.net.forward(inputs, int_act_fn, out_act_fn)
        __, predicted_class = torch.max(fwd_outputs.detach(), 1)  # first return value is the max value, second is argmax

        out_data = fwd_outputs.detach().cpu().numpy()
        pred = predicted_class.cpu().numpy()

        return out_data, pred

    def predict_prob(self, testloader, int_act_fn = 'relu', out_act_fn='softmax'):
        '''
        Returns probability of output classes given dataloader
        :param testloader: Dataloader which returns test features
        :param int_act_fn: activation function for hidden layers
        :param out_act_fn: activation function for output layers
        :return: 2D array (n_inst, n_classes) with probability of each class for each instance
        '''

        out_probs = list()

        for inputs in testloader:

            inputs = inputs.to(self.device)

            out_data, __ = self._get_prediction(inputs, int_act_fn, out_act_fn)

            for i, row in enumerate(out_data):
                out_probs.append(row)

        return np.vstack(out_probs)


    def predict_prob_from_array(self, x_test, int_act_fn = 'relu', out_act_fn='softmax'):
        '''
        Get prediction probability given test features as array instead of dataloader object
        :param x_test: 2D array of test feats (n_inst, n_feats)
        :param int_act_fn: activation function for hidden layers
        :param out_act_fn: activation function for output layers
        :param cuda: True if the device has cuda
        :return: 2D array of prediction probs (n_inst * n_classes)
        '''

        inputs = torch.from_numpy(x_test).float()

        inputs = inputs.to(self.device)

        return self.net.forward(inputs, int_act_fn, out_act_fn).detach().cpu().numpy()




