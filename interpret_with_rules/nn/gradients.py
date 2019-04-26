from torch.autograd import Variable
import torch

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import maxabs_scale, minmax_scale

import math

def get_out_gradient(net, dataloader, int_act_fn = 'relu', output_act_fn='softmax', grad_type = 'label', cuda = torch.cuda.is_available()):
    '''
    Get gradients of output nodes wrt every input node for every instance.
    :param net: Instance of trained neural network
    :param dataloader: data loader
    :param int_act_fn: activation function for hidden layers in the forward pass
    :param output_act_fn: activation function for output layer in the forward pass
    :param grad_type: (max | sum) whether to return max gradient across all output nodes, or sum of gradients at all output nodes
    :param cuda: True if GPU is supported
    :return: scipy sparse csr matrix with total gradient of output nodes w.r.t input node for every input node and instance
    '''
    net.cuda() if cuda else net

    gradients = list()

    for idx, (inputs, labels) in enumerate(dataloader):

        inputs = inputs.cuda() if cuda else inputs
        inputs = Variable(inputs, requires_grad=True)

        net.zero_grad()
        fwd_out = net.forward(inputs, int_act_fn, output_act_fn)

        if grad_type == 'sum':
            cur_grad = get_total_grad_out(inputs, fwd_out)
        elif grad_type == 'max':
            cur_grad = get_max_grad_out(inputs, fwd_out)
        elif grad_type == 'argmax':
            cur_grad = get_argmax_out_grad(inputs, fwd_out)
        elif grad_type == 'label':
            cur_grad = get_label_grad(inputs, fwd_out, labels)
        else:
            raise ValueError('Please enter correct value for grad_type (sum|max|argmax|label)')

        gradients.append(cur_grad)
        inputs.grad.detach_()  # very important to prevent mem leak

    gradients = np.vstack(gradients)

    print("Number of zero gradients: ", (gradients.shape[0] * gradients.shape[1]) - np.count_nonzero(gradients))
    #print("Zero gradient found at rows and columns (respectively): ", np.where(gradients == 0.))
    
    #gradients = scale_grads(gradients)

    #print("Number of zero gradients after rescaling: ", (gradients.shape[0] * gradients.shape[1]) - np.count_nonzero(gradients))

    gradients = csr_matrix(gradients)

    return gradients

def get_max_rms_gradient(net, dataloader, int_act_fn ='relu', output_act_fn='softmax', cuda = torch.cuda.is_available()):
    '''
    Get sensitivity of every feature based on trained neural network.
    The implementation is based on the following work:
    Engelbrecht, A., and I. Cloete.
    "Feature extraction from feedforward neural networks using sensitivity analysis."
    Proceedings of the International Conference on Systems, Signals, Control, Computers. 1998.
    :param net: Instance of trained neural network
    :param dataloader: data loader
    :param int_act_fn: activation function for hidden layers in the forward pass
    :param output_act_fn: activation function for output layer in the forward pass
    :param cuda: True if GPU is supported
    :return: scipy sparse csr matrix with total gradient of output nodes w.r.t input node for every input node and instance
    :return: 1D numpy array with indices of features in decreasing order of sensitivity
    '''
    net.cuda() if cuda else net

    gradients = list()

    for idx, inputs in enumerate(dataloader):
        inputs = inputs.cuda() if cuda else inputs
        inputs = Variable(inputs, requires_grad=True)

        net.zero_grad()
        fwd_out = net.forward(inputs, int_act_fn, output_act_fn)

        cur_grads = get_all_out_grads(inputs, fwd_out)

        gradients.append(cur_grads)

        inputs.grad.detach_()  # very important to prevent mem leak

    gradients = torch.cat(gradients, dim = 1)
    gradients = get_rms_all_out(gradients)
    gradients, __ = get_max_feat_grad(gradients)

    __, max_grad_idx = torch.topk(gradients, k = gradients.shape[0])
    max_grad_idx = max_grad_idx.cpu().numpy()

    gradients = csr_matrix(gradients.cpu().numpy())

    return gradients, max_grad_idx

def get_all_out_grads(inputs, fwd_out):
    """
    Get squared gradient of output nodes w.r.t input nodes for all instances
    :param inputs: Input, batch_size * n_feats
    :param fwd_out: Output, batch_size * n_out
    :return: 3D array of gradients, shape: n_out_nodes * n_inst * n_feats
    """
    grads = torch.zeros(fwd_out.shape[1], inputs.shape[0], inputs.shape[1]) #shape: n_out_nodes * n_inst * n_feats

    for i in range(fwd_out.shape[1]):
        cur_grad_tensor = torch.zeros_like(fwd_out)
        cur_grad_tensor[:,i] = 1

        fwd_out.backward(cur_grad_tensor, retain_graph=True)
        grad = inputs.grad.data
        grad = grad.pow(2)

        grads[i] = grad

        inputs.grad.detach_() #IMP!
        inputs.grad.zero_() #IMP!
        
    return grads

def get_rms_all_out(grads):
    """
    Get root mean square value of gradients of every feature for every output node across all instances
    :param grads: 3D array of squared gradients, shape: n_out_nodes * n_inst * n_feats
    :return: 2D array (n_out * n_feats) for root mean square value of gradients across all instances for all output node, feature pairs
    """

    rms_grads = torch.zeros(grads.shape[0], grads.shape[2])

    for i in range(grads.shape[0]):
        rms_grads[i] = torch.sqrt(torch.mean(grads[i], dim = 0))
        # print("Shape of new grad vector:", rms_grads[i].shape)

    return rms_grads

def get_max_feat_grad(grads):
    '''
    Return the maximum significance of a feature across all output nodes
    :param grads: 2D array with significance scores (RMS gradients across instances) (n_out * n_feats)
    :return: 1D array of maximum feature significance across all output nodes
    '''
    return torch.max(grads, dim = 0)

def get_total_grad_out(inputs, fwd_out):
    '''
    :param inputs: network inputs
    :param fwd_out: network output after forward pass
    :return: Sum of gradients of all output nodes wrt input nodes
    '''
    # in the following function, torch.ones of the same shape as fwd_out indicates that gradient should be calculated wrt all the entries of fwd_out.
    # it calculates gradients for all non-zero entries. If we want gradients for only one entry, we can make only that val 1, and the rest 0.
    # further, if gradients for multiple cols (dimensions) of fwd_out are computed, they are added together when we do inputs.grad.
    # If we don't want gradients to be added together, we need to iterate over all columns and zero out all but one column seqeuntially.
    # fwd_out.backward(torch.ones(fwd_out.size()).cuda())
    fwd_out.backward(torch.ones_like(fwd_out))
    return inputs.grad.data.cpu().numpy()


def get_max_grad_out(inputs, fwd_out):
    '''
    Returns the maximum value (across all output nodes) of gradient of output w.r.t input
    :param inputs: network inputs
    :param fwd_out: network output after forward pass
    :return: Maximum gradient for the inputs across all output nodes
    '''

    prev_max = None
    for i in range(fwd_out.shape[1]):
        cur_grad_tensor = torch.zeros_like(fwd_out)
        cur_grad_tensor[:,i] = 1

        fwd_out.backward(cur_grad_tensor)

        if prev_max is None:
            prev_max = inputs.grad.data
        else:
            prev_max = torch.max(prev_max.abs(), inputs.grad.data.abs())

        inputs.grad.detach_() #IMP!

    return prev_max.cpu().numpy()

def get_argmax_out_grad(inputs, fwd_out):
    '''
    :param inputs: network inputs
    :param fwd_out: network output after forward pass
    :return Gradient of output w.r.t input for the output node with the highest value for all instances
    '''
    __, argmax = torch.max(fwd_out, dim = 1)
    argmax = argmax.data

    return get_label_grad(inputs, fwd_out, argmax)

def get_label_grad(inputs, fwd_out, label_node):
    '''
    :param inputs: network inputs
    :param fwd_out: network output after forward pass
    :param label_node: 1D array of output label nodes to compute gradient of.
    :return: Gradient of output w.r.t input for the given output nodes for all instances
    '''
    grad_tensor = torch.zeros_like(fwd_out)

    label_node = label_node.cpu().numpy()

    grad_tensor[np.arange(len(label_node)), label_node] = 1

    fwd_out.backward(grad_tensor)

    grads = inputs.grad.data

    return grads.cpu().numpy()


def scale_grads(gradients):
    """
    #Scales the absolute gradients for features for each instance between 0 and 1 and preserves the sign
    :param gradients: Gradients
    :return: Rescaled gradients
    """
    return np.sign(gradients) * minmax_scale(np.abs(gradients), axis = 1)
