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
import random
from src.tmodel import *
from src.amodel import *
from src.utils import *


class Target:

    def __init__(self, config):
        self.config = config
        self.gpu = False

    def load_parameter(self):
        # read parameter from file
        with open(self.config) as json_file:
            self.parameter = json.load(json_file)
            self.gpu = True if self.parameter['gpu'] > 0 else False

    def load_dataset(self, dataset):
        # create new Dataset object
        self.dataset = Dataset(dataset[0],
                               dataset[0].ndata['feat'],
                               dataset[0].ndata['label'],
                               dataset[0].ndata['feat'].shape[1],
                               dataset.num_classes)
        self.dataset.generate_masks(0.2, 0.4, 0.4)

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

        # load model to gpu
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
            t0 = time.time()

            # forward
            logits = self.model(self.dataset.graph, self.dataset.features)
            loss = F.cross_entropy(logits[self.dataset.train_nid], self.dataset.labels[self.dataset.train_nid])

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
            # query model
            logits = self.model(self.dataset.graph, self.dataset.features)
            logits = logits[nid]
            labels = self.dataset.labels[nid]
            _, indices = torch.max(logits, dim=1)
            # calculate accuracy
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

    def get_posteriors(self, nid):
        self.model.eval()
        with torch.no_grad():
            # query model
            logits = self.model(self.dataset.graph, self.dataset.features)
            logits = logits[nid]
            # return posteriors predicted by the model
            return logits


class Attacker:

    def __init__(self, config, target):
        self.config = config
        self.target_model = target
        self.gpu = False

    def load_parameter(self):
        # read parameter from file
        with open(self.config) as json_file:
            self.parameter = json.load(json_file)
            self.gpu = True if self.parameter['gpu'] > 0 else False

    def create_dataset(self, mdata, nid=None, create_nid=True):
        if create_nid:
            # overwrite nid to use the full dataset
            attacker_mask = torch.ones(mdata[0].number_of_nodes(), dtype=torch.bool)
            nid = attacker_mask.nonzero().squeeze()

        # number of nodes in nid
        n_nodes = nid.shape[0]

        # feature amount
        self.num_features = self.target_model.get_posteriors([0]).shape[1] * 2
        # input features and labels
        self.features = torch.zeros((n_nodes, self.num_features), dtype=torch.float)
        self.labels = torch.zeros(n_nodes, dtype=torch.long)

        # create node pairs
        for i in range(1, n_nodes, 2):
            # query target model to get posteriors
            post_i = self.target_model.get_posteriors([nid[i - 1].item()])
            post_j = self.target_model.get_posteriors([nid[i].item()])
            feature = torch.cat((post_i, post_j), 1)
            # use original datasat.graph to obtain the label
            label = mdata[0].has_edge_between(nid[i - 1], nid[i])
            self.features[i] = feature
            self.labels[i] = torch.ones(1, 1) if label else torch.zeros(1, 1)

        # train, eval, test : 10%, 45%, 45%
        self.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_val_split = int(n_nodes * 0.1)
        val_test_split  = train_val_split + int(n_nodes * 0.45)

        # train masks
        for a in range(n_nodes):
            self.train_mask[a] = a < train_val_split
            self.val_mask[a] = a >= train_val_split and a < val_test_split
            self.test_mask[a] = a >= val_test_split

        # node ids
        self.train_nid = self.train_mask.nonzero().squeeze()
        self.val_nid = self.val_mask.nonzero().squeeze()
        self.test_nid = self.test_mask.nonzero().squeeze()

    def initialize(self):
        # create model
        self.model = FNN(self.num_features,
                         self.parameter['n_hidden'],
                         2,
                         self.parameter['n_layers'],
                         F.relu,
                         self.parameter['dropout'])

        # load model to gpu
        if self.gpu:
            torch.cuda.set_device(self.parameter['gpu'])
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
            t0 = time.time()

            # forward
            logits = self.model(self.features)
            loss = F.cross_entropy(logits[self.train_nid], self.labels[self.train_nid])

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            dur.append(time.time() - t0)

            # evaluate
            acc = self.evaluate(self.val_nid)
            if show_process:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f}".format(epoch, np.mean(dur), loss.item(), acc))

    def evaluate(self, nid):
        self.model.eval()
        with torch.no_grad():
            # query model
            logits = self.model(self.features)
            logits = logits[nid]
            labels = self.labels[nid]
            _, indices = torch.max(logits, dim=1)
            # calculate accuracy
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

    def query(self, i, j):
        self.model.eval()
        # query target model to get posteriors of nodes i and j
        post_i = self.target_model.get_posteriors([i])
        post_j = self.target_model.get_posteriors([j])
        # generate input feature for attacker model
        feature = torch.cat((post_i, post_j), 1)
        with torch.no_grad():
            # query model on input feature
            logits = self.model(feature)
            _, indices = torch.max(logits, dim=1)
        # evaluate posteriors of model
        return True if indices == 1 else False



def main(dset, vv=False, vvv=False):
    # seed
    random.seed(1234)

    # main dataset
    mdata = load_data(dset)

    # target
    target = Target('config/target-model.conf')
    target.load_parameter()
    target.load_dataset(mdata)
    target.initialize()
    target.train(show_process=vvv)
    target_acc = target.evaluate(target.dataset.test_nid)

    # attacker
    attacker = Attacker('config/attacker-model.conf', target)
    attacker.load_parameter()
    attacker.create_dataset(mdata, target.dataset.test_nid, create_nid=False)
    attacker.initialize()
    attacker.train(show_process=vvv)
    attacker_acc = attacker.evaluate(attacker.test_nid)

    # print results if verbose
    if vv:
        print("\n [ Target model ]\n\n\tType: GraphSAGE\n\tAccuracy: {}\n".format(target_acc))
        print("\n [ Attacker model ]\n\n\tType: FNN\n\tAccuracy: {}\n".format(attacker_acc))

    # return target_accuracy and attacker_accuracy for evaluation
    return target_acc, attacker_acc


# Link Stealing Attack
if __name__ == '__main__':
    # cmd args
    parser = argparse.ArgumentParser(description='Link-Stealing Attack')
    register_data_args(parser)
    args = parser.parse_args()

    main(args.dataset, vv=True, vvv=False)
