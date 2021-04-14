# Python 3.8.5

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Imports
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import json
from src.tmodel import *
from src.amodel import *
from src.utils import *


class Target:

    def __init__(self, config):
        self.config = config
        self.parameter = {}
        self.dataset = None
        self.model = None
        self.optimizer = None
        self.gpu = False

    def load_parameter(self):
        with open(self.config) as json_file:
            self.parameter = json.load(json_file)
            self.gpu = True if self.parameter['gpu'] > 0 else False

    def load_dataset(self, mdata):
        self.dataset = Dataset(
                            mdata[0],                        # graph
                            mdata[0].ndata['feat'],          # features
                            mdata[0].ndata['label'],         # labels
                            mdata[0].ndata['train_mask'],    # train mask
                            mdata[0].ndata['val_mask'],      # val mask
                            mdata[0].ndata['test_mask'],     # test mask
                            mdata[0].ndata['feat'].shape[1], # amount of features
                            mdata.num_classes,               # amount of classes
                            mdata.graph.number_of_edges)     # amount of edges


    def initialize(self):
        # GPU
        if self.gpu:
            self.dataset.to(self.parameter['gpu'])

        # preprocess dataset
        self.dataset.preprocess_data(self.parameter['gpu'])

        # create model
        self.model = GraphSAGE(
                            self.dataset.n_features,
                            self.parameter['n_hidden'],
                            self.dataset.n_classes,
                            self.parameter['n_layers'],
                            F.relu,
                            self.parameter['dropout'],
                            self.parameter['aggregator_type'])

        if self.gpu:
            self.model.cuda()

        # optimizer
        self.optimizer = torch.optim.Adam(
                                        self.model.parameters(),
                                        lr=self.parameter['lr'],
                                        weight_decay=self.parameter['weight_decay'])

    def train(self, show_process=False):
        dur = []
        for epoch in range(self.parameter['n_epochs']):
            self.model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits = self.model(self.dataset.graph, self.dataset.features)
            loss = F.cross_entropy(logits[self.dataset.train_nid], self.dataset.labels[self.dataset.train_nid])

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            # evaluate
            acc = self.evaluate(self.dataset.val_nid)
            if show_process:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                      "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                                    acc, self.dataset.n_edges / np.mean(dur) / 1000))

    def evaluate(self, nid):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.dataset.graph, self.dataset.features)
            logits = logits[nid]
            labels = self.dataset.labels[nid]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

    def get_posteriors(self, nid):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.dataset.graph, self.dataset.features)
            logits = logits[nid]
            return logits


if __name__ == '__main__':

    # cmd args
    parser = argparse.ArgumentParser(description='Link-Stealing Attack')
    register_data_args(parser)
    args = parser.parse_args()

    # main dataset
    dataset = load_data(args)

    # target
    target = Target('config/target-model.conf')
    target.load_parameter()
    target.load_dataset(dataset)
    target.initialize()
    target.train(show_process=False)
    print("\n [ Target model ]\n\n\tType: GraphSAGE\n\tAccuracy: {}\n".format(target.evaluate(target.dataset.test_nid)))
