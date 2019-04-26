import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class FFNNet(nn.Module):

    def __init__(self, n_hid_layers, n_in, n_out, n_hid = None):

        super(FFNNet, self).__init__()

        if n_hid_layers is None:
            raise ValueError("Please enter valid number of hidden layers")
        elif n_hid_layers != 0 and n_hid is None:
            raise ValueError("Please enter valid number of hidden nodes")

        self.n_hid_layers = n_hid_layers
        self._construct_layers(n_in, n_out, n_hid)

    def _construct_layers(self, n_in, n_out, n_hid):

        layers = []

        if not self.n_hid_layers:
            layers.append(nn.Linear(n_in, n_out))
        else:
            for i, __ in enumerate(range(self.n_hid_layers)):
                if not i:
                    layers.append(nn.Linear(n_in, n_hid)) #input layer
                else:
                    layers.append(nn.Linear(n_hid, n_hid)) #hidden layers

            layers.append(nn.Linear(n_hid, n_out)) #output layer

        self.layers = nn.ModuleList(layers)

    def forward(self, x, int_act_fn = 'relu', output_act_fn='softmax'):
        '''
        Performs forward pass on a neural network
        :param x: input feats (batch)
        :param int_act_fn: activation function for hidden layers
        :param output_act_fn: activation function for output layer
        :return: Output after forward pass of neural network on a given input
        '''

        for i, cur_layer in enumerate(self.layers):
            if i is (len(self.layers) - 1):
                x = getattr(F, output_act_fn)(cur_layer(x))
            else:
                x = getattr(F, int_act_fn)(cur_layer(x))

        return x

    def backprop(self, fwd_out, target, loss = 'CrossEntropyLoss', optimizer_name ='Adam'):
        '''
        Perform backpropagation
        :param fwd_out: output of the forward pass
        :param target: output labels
        :param loss: name of loss function
        :param optimizer_name: name of optimizer
        :return: loss object after finishing backward pass
        '''

        loss_fn = getattr(nn, loss)() #define loss function
        loss = loss_fn(fwd_out, target) #calculate loss

        optimizer = getattr(optim, optimizer_name)(self.parameters())

        optimizer.zero_grad() #reset tensor gradients
        loss.backward()  #compute gradients for network params w.r.t loss
        optimizer.step() #perform the gradient update step

        return loss