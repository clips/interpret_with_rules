from ffnn import FFNNet
from dataset import SparseDataset, CustomDataset
from utils import get_dataloader, save_model, load_model
from execute import train, predict

import sys
sys.path.append("..")

from newsgroups.process_data import get_featurized_data

import os

from scipy.sparse import csr_matrix
import numpy as np
import torch
# from torch.autograd import Variable

def main():
    net = FFNNet(2, 5000, 20, 100)

    # x = Variable(torch.randn(20, 5000))
    # y = Variable(torch.arange(0, 20, 1)).long()

    #fwd_out = net.forward(x)

    #net.backprop(fwd_out, y)
    # print(net.layers[0].bias.grad)

    # x = torch.randn(20, 5000)
    x = csr_matrix((20, 5000), dtype=np.float32)

    y = torch.arange(0, 20, 1).long()

    ds = SparseDataset(x, y)
    # ds = CustomDataset(x, y)
    trainloader = get_dataloader(ds, shuffle=True)

    train(net, trainloader)

    out_dir = '../../out/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    torch.save(net, out_dir+'torch_model_test.pkl')

    testloader = get_dataloader(ds, shuffle=False)
    pred_class, pred_probs = predict(net, testloader)
    print("Predicted classes ", pred_class)


if __name__ == '__main__':
    main()