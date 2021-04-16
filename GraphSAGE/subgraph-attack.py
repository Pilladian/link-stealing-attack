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
import copy
from src.tmodel import *
from src.amodel import *
from src.utils import *


class Target:

    def __init__(self, config, traingraph, num_classes):
        self.config = config
        self.gpu = False
        self.traingraph = traingraph
        self.num_classes = num_classes

    def load_parameter(self):
        # read parameter from file
        with open(self.config) as json_file:
            self.parameter = json.load(json_file)
            self.gpu = True if self.parameter['gpu'] > 0 else False

    def initialize(self):
        # GPU
        if self.gpu:
            torch.cuda.set_device(gpu)
            self.traingraph.cuda()
            self.evalgraph.cuda()

        # create model
        self.model = GraphSAGE(
                            self.traingraph.ndata['feat'].shape[1],
                            self.parameter['n_hidden'],
                            self.num_classes,
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

    def train(self, verbose=False):
        for epoch in range(self.parameter['n_epochs']):
            self.model.train()
            # forward
            logits = self.model(self.traingraph, self.traingraph.ndata['feat'])
            loss = F.cross_entropy(logits, self.traingraph.ndata['label'])
            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose:
                print(f'Target Training: {epoch + 1} / {self.parameter["n_epochs"]}')

    def evaluate(self, graph):
        if self.gpu:
            graph.cuda()

        self.model.eval()
        with torch.no_grad():
            # query model
            logits = self.model(graph, graph.ndata['feat'])
            logits = logits
            labels = graph.ndata['label']
            _, indices = torch.max(logits, dim=1)
            # calculate accuracy
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

    def get_posteriors(self, graph, id):
        if self.gpu:
            graph.cuda()

        self.model.eval()
        with torch.no_grad():
            # query model
            logits = self.model(graph, graph.ndata['feat'])
            logits = logits[id]
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

    def sample_dataset(self, graph):
        SIZE = int(graph.num_edges() * 0.8)
        # modified graph
        self.modified_graph = copy.deepcopy(graph)

        # edges in modified_graph that have been in graph but are going to be deleted
        pos_samples = []
        # edges that do not exist in (modified_)graph
        neg_samples = []

        # pos_samples
        for p in range(int(SIZE / 2)):
            edge_id = random.randint(0, self.modified_graph.num_edges() - 1)
            src, dst = self.modified_graph.find_edges([edge_id])[0].item(), self.modified_graph.find_edges([edge_id])[1].item()
            self.modified_graph.remove_edges([self.modified_graph.edge_id(src, dst)])
            pos_samples.append(((src, dst), True))

        # neg_samples
        for p in range(int(SIZE / 2)):
            src, dst = random.randint(0, graph.num_nodes() - 1), random.randint(0, graph.num_nodes() - 1)
            while graph.has_edges_between(src, dst) and (src, dst) not in neg_samples:
                src, dst = random.randint(0, graph.num_nodes() - 1), random.randint(0, graph.num_nodes() - 1)
            neg_samples.append(((src, dst), False))

        # create dataset
        samples = pos_samples + neg_samples
        random.shuffle(samples)
        self.feature_amount = self.target_model.get_posteriors(self.modified_graph, 0).shape[0] * 2
        self.features = torch.zeros((SIZE, self.feature_amount), dtype=torch.float)
        self.labels = torch.zeros(SIZE, dtype=torch.long)

        for i, ((src, dst), label) in enumerate(samples):
            # query target model to get posteriors
            post_src = self.target_model.get_posteriors(self.modified_graph, src)
            post_dst = self.target_model.get_posteriors(self.modified_graph, dst)
            feature = torch.cat((post_src, post_dst))
            # use original datasat.graph to obtain the label
            self.features[i] = feature
            self.labels[i] = torch.ones(1, 1) if label else torch.zeros(1, 1)

        # train, eval, test : 20%, 40%, 40%
        self.train_mask = torch.zeros(SIZE, dtype=torch.bool)
        self.val_mask = torch.zeros(SIZE, dtype=torch.bool)
        self.test_mask = torch.zeros(SIZE, dtype=torch.bool)

        train_val_split = int(SIZE * 0.2)
        val_test_split  = train_val_split + int(SIZE * 0.4)

        # train masks
        for a in range(SIZE):
            self.train_mask[a] = a < train_val_split
            self.val_mask[a] = a >= train_val_split and a < val_test_split
            self.test_mask[a] = a >= val_test_split

        # node ids
        self.train_nid = self.train_mask.nonzero().squeeze()
        self.val_nid = self.val_mask.nonzero().squeeze()
        self.test_nid = self.test_mask.nonzero().squeeze()

    def initialize(self):
        # create model
        self.model = FNN(self.feature_amount,
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

    def train(self, verbose=False):
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
            if verbose:
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

    def query(self, graph, i, j):
        self.model.eval()
        # query target model to get posteriors of nodes i and j
        post_i = self.target_model.get_posteriors(graph, i)
        post_j = self.target_model.get_posteriors(graph, j)
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

    # split main graph / dataset in train and test graph
    split = mdata[0].number_of_nodes() * 0.5
    train_mask = torch.zeros(mdata[0].number_of_nodes(), dtype=torch.bool)
    test_mask = torch.zeros(mdata[0].number_of_nodes(), dtype=torch.bool)

    for a in range(mdata[0].number_of_nodes()):
        train_mask[a] = a < split
        test_mask[a] = a >= split

    traingraph = mdata[0].subgraph(mdata[0].ndata['train_mask'])
    testgraph = mdata[0].subgraph(mdata[0].ndata['test_mask'])

    # remove self loops
    traingraph = dgl.remove_self_loop(traingraph)
    testgraph = dgl.remove_self_loop(testgraph)

    # target
    target = Target('config/target-model.conf', traingraph, mdata.num_classes)
    target.load_parameter()
    target.initialize()
    target.train(verbose=vvv)

    # attacker
    attacker = Attacker('config/attacker-model.conf', target)
    attacker.load_parameter()
    attacker.sample_dataset(testgraph)
    attacker.initialize()
    attacker.train(verbose=vvv)

    # evaluation
    target_acc = target.evaluate(attacker.modified_graph)
    attacker_acc = attacker.evaluate(attacker.test_nid)

    # print results if verbose
    if vv:
        print(f"\n [ Target model ]\n\n\tType: GraphSAGE\n\tAccuracy: {target_acc:.2f}\n")
        print(f"\n [ Attacker model ]\n\n\tType: FNN\n\tAccuracy: {attacker_acc:.2f}\n")

    # return target_accuracy and attacker_accuracy for evaluation
    return target_acc, attacker_acc


# Link Stealing Attack
if __name__ == '__main__':
    # cmd args
    parser = argparse.ArgumentParser(description='Link-Stealing Attack')
    register_data_args(parser)
    args = parser.parse_args()

    main(args.dataset, vv=True, vvv=False)
