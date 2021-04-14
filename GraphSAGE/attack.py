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


class Attacker:

    def __init__(self, config, target):
        self.config = config
        self.parameter = {}
        self.target_model = target
        self.model = None
        self.optimizer = None
        self.gpu = False

    def load_parameter(self):
        with open(self.config) as json_file:
            self.parameter = json.load(json_file)
            self.gpu = True if self.parameter['gpu'] > 0 else False

    def create_dataset(self, mdata):
        # graph
        graph = mdata[0]

        # set of all eval nodes of target model
        targets_test_nid = self.target_model.dataset.test_nid
        n_nodes = targets_test_nid.shape[0]

        # feature amount
        self.n_features = self.target_model.get_posteriors([0]).shape[1] * 2
        # input features and labels
        self.features = torch.zeros((n_nodes, self.n_features), dtype=torch.float)
        self.labels = torch.zeros(n_nodes, dtype=torch.long)

        for i, node in enumerate(targets_test_nid):
            # neighbors -> label = True
            if bool(random.getrandbits(1)):
                poss_neighbors = graph.out_edges(i)
                neighbor_amount = poss_neighbors[1].shape[0]
                # has no neighbors -> label = False
                if neighbor_amount < 1:
                    neighbor_id = i - 1
                    post_i = self.target_model.get_posteriors([i])
                    post_neighbor = self.target_model.get_posteriors([neighbor_id])
                    feature = torch.cat((post_i, post_neighbor), 1)
                    self.features[i] = feature
                    self.labels[i] = torch.zeros(1, 1)

                # has at least one neighbor -> label True
                else:
                    neighbor_id = random.randint(0, neighbor_amount - 1)
                    post_i = self.target_model.get_posteriors([i])
                    post_neighbor = self.target_model.get_posteriors([neighbor_id])
                    feature = torch.cat((post_i, post_neighbor), 1)
                    self.features[i] = feature
                    self.labels[i] = torch.ones(1, 1)

            # no neighbors -> label = False
            else:
                poss_neighbors = graph.out_edges(i)
                neighbor_amount = poss_neighbors[1].shape[0]
                # has no neighbors -> label = False
                if neighbor_amount < 1:
                    neighbor_id = i - 1
                # has neighbors -> node that is no neighbor -> label = False
                else:
                    neighbor_id = random.randint(0, n_nodes - 1)
                    while neighbor_id == i or neighbor_id in poss_neighbors[1].tolist():
                        neighbor_id = random.randint(0, n_nodes - 1)

                post_i = self.target_model.get_posteriors([i])
                post_neighbor = self.target_model.get_posteriors([neighbor_id])
                feature = torch.cat((post_i, post_neighbor), 1)
                self.features[i] = feature
                self.labels[i] = torch.zeros(1, 1)

        # train, eval, test : 20%, 40%, 40%
        self.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_val_split = int(n_nodes * 0.2)
        val_test_split  = train_val_split + int(n_nodes * 0.4)

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
        self.model = FNN(self.n_features,
                         self.parameter['n_hidden'],
                         2,
                         self.parameter['n_layers'],
                         F.relu,
                         self.parameter['dropout'])

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
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits = self.model(self.features)
            loss = F.cross_entropy(logits[self.train_nid], self.labels[self.train_nid])

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            # evaluate
            acc = self.evaluate(self.val_nid)
            if show_process:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f}".format(epoch, np.mean(dur), loss.item(), acc))

    def evaluate(self, nid):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.features)
            logits = logits[nid]
            labels = self.labels[nid]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)


def main(dset, v=False, vvv=False):
    # seed
    random.seed(1234)

    # main dataset
    dataset = load_data(dset)

    # target
    target = Target('config/target-model.conf')
    target.load_parameter()
    target.load_dataset(dataset)
    target.initialize()
    target.train(show_process=vvv)
    target_acc = target.evaluate(target.dataset.test_nid)

    # attacker
    attacker = Attacker('config/attacker-model.conf', target)
    attacker.load_parameter()
    attacker.create_dataset(dataset)
    attacker.initialize()
    attacker.train(show_process=vvv)
    attacker_acc = attacker.evaluate(attacker.test_nid)

    # print results if verbose
    if v:
        print("\n [ Target model ]\n\n\tType: GraphSAGE\n\tAccuracy: {}\n".format(target_acc))
        print("\n [ Attacker model ]\n\n\tType: FNN\n\tAccuracy: {}\n".format(attacker_acc))

    return target_acc, attacker_acc


# Link Stealing Attack
if __name__ == '__main__':
    # cmd args
    parser = argparse.ArgumentParser(description='Link-Stealing Attack')
    register_data_args(parser)
    args = parser.parse_args()

    main(args.dataset, v=True, vvv=False)
