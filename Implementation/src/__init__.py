# Python 3.8.5

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Imports
import copy
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
from src.graphsage import *
from src.mlp import *
from src.gat import *
from src.gcn import *
import time
from scipy.spatial import distance

def _stats(pred, labels):
    pos_mask = labels == 1
    neg_mask = labels == 0

    tp = torch.sum(pred[pos_mask] == labels[pos_mask])
    tn = torch.sum(pred[neg_mask] == labels[neg_mask])

    fp = torch.sum(pred[pos_mask] != labels[pos_mask])
    fn = torch.sum(pred[neg_mask] != labels[neg_mask])

    return tp, tn, fp, fn


def _precision(pred, labels):
    tp, tn, fp, fn = _stats(pred, labels)
    return tp / (tp + fp)

def _recall(pred, labels):
    tp, tn, fp, fn = _stats(pred, labels)
    return tp / (tp + fn)

def _f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def _accuracy(pred, labels):
    tp, tn, fp, fn = _stats(pred, labels)
    return (tp + tn) / (tp + tn + fp + fn)


class Target:

    def __init__(self, gnn, graph, num_classes):
        self.gnn_name = gnn
        self.graph = copy.deepcopy(graph)
        self.num_classes = num_classes
        self._load_parameter()

    def _load_parameter(self):
        # read parameter from file
        with open(f'config/{self.gnn_name}.conf') as json_file:
            self.parameter = json.load(json_file)
            self.gpu = True if self.parameter['gpu'] > 0 else False

    def train(self, verbose=False):
        # initialize model
        self._initialize()

        for epoch in range(self.parameter['n_epochs']):
            self.model.train()
            # forward
            if self.gnn_name == 'graphsage':
                logits = self.model(self.graph, self.graph.ndata['feat'])
            elif self.gnn_name in ['gat', 'gcn']:
                g = dgl.remove_self_loop(self.graph)
                self.model.g = dgl.add_self_loop(g)
                logits = self.model(self.graph.ndata['feat'])
            loss = F.cross_entropy(logits, self.graph.ndata['label'])
            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose:
                print(f'Target Training: {epoch + 1} / {self.parameter["n_epochs"]}')

    def _initialize(self):
        # GPU
        if self.gpu:
            torch.cuda.set_device(gpu)
            self.traingraph.cuda()

        # create model
        if self.gnn_name == 'graphsage':
            self.model = GraphSAGE(
                            self.graph.ndata['feat'].shape[1],
                            self.parameter['n_hidden'],
                            self.num_classes,
                            self.parameter['n_layers'],
                            F.relu,
                            self.parameter['dropout'],
                            self.parameter['aggregator_type'])

        elif self.gnn_name == 'gat':
            heads = ([self.parameter['n_heads']] * self.parameter['n_layers']) + [self.parameter['n_outheads']]
            self.model = GAT(
                            self.graph,
                            self.parameter['n_layers'],
                            self.graph.ndata['feat'].shape[1],
                            self.parameter['n_hidden'],
                            self.num_classes,
                            heads,
                            F.elu,
                            self.parameter['dropout'],
                            self.parameter['dropout'],
                            self.parameter['negative_slope'],
                            self.parameter['residual'])

        elif self.gnn_name == 'gcn':
            self.model = GCN(
                            self.graph,
                            self.graph.ndata['feat'].shape[1],
                            self.parameter['n_hidden'],
                            self.num_classes,
                            self.parameter['n_layers'],
                            F.relu,
                            self.parameter['dropout'])

        # load model to gpu
        if self.gpu:
            self.model.cuda()

        # optimizer
        self.optimizer = torch.optim.Adam(
                                        self.model.parameters(),
                                        lr=self.parameter['lr'],
                                        weight_decay=self.parameter['weight_decay'])

    def evaluate(self, graph):
        if self.gpu:
            graph.cuda()

        self.model.eval()
        with torch.no_grad():
            # query model
            if self.gnn_name == 'graphsage':
                logits = self.model(graph, graph.ndata['feat'])
            elif self.gnn_name in ['gat', 'gcn']:
                g = dgl.remove_self_loop(graph)
                self.model.g = dgl.add_self_loop(g)
                logits = self.model(graph.ndata['feat'])
            labels = graph.ndata['label']
            _, pred = torch.max(logits, dim=1)

            # accuracy
            return _accuracy(pred, labels)

    def get_posteriors(self, graph, id):
        if self.gpu:
            graph.cuda()

        self.model.eval()
        with torch.no_grad():
            # query model
            if self.gnn_name == 'graphsage':
                logits = self.model(graph, graph.ndata['feat'])
            elif self.gnn_name in ['gat', 'gcn']:
                g = dgl.remove_self_loop(graph)
                self.model.g = dgl.add_self_loop(g)
                logits = self.model(graph.ndata['feat'])
            logits = logits[id]
            # return posteriors predicted by the model
            return logits


class Attacker:

    def __init__(self, target, graph):
        self.target_model = target
        self.graph = copy.deepcopy(graph)
        self._load_parameter()

    def _load_parameter(self):
        # read parameter from file
        with open(f'config/mlp.conf') as json_file:
            self.parameter = json.load(json_file)
            self.gpu = True if self.parameter['gpu'] > 0 else False

    def create_modified_graph(self, survivors):
        # modified graph
        self.modified_graph = dgl.remove_self_loop(copy.deepcopy(self.graph))
        orig_num_of_edges = self.modified_graph.number_of_edges()

        pos = []
        for p in range(int(orig_num_of_edges * (1 - survivors))):
            edge_id = random.randint(0, self.modified_graph.num_edges() - 1)
            src, dst = self.modified_graph.find_edges([edge_id])[0].item(), self.modified_graph.find_edges([edge_id])[1].item()
            self.modified_graph.remove_edges([self.modified_graph.edge_id(src, dst)])
            pos.append(((src, dst), True))

        for p in range(int(orig_num_of_edges * survivors)):
            edge_id = random.randint(0, self.modified_graph.num_edges() - 1)
            src, dst = self.modified_graph.find_edges([edge_id])[0].item(), self.modified_graph.find_edges([edge_id])[1].item()
            pos.append(((src, dst), True))

        # neg_samples - edges that do not exist in (modified_)graph
        neg = []
        for n in range(int(orig_num_of_edges)):# * (1 - survivors))):
            src, dst = random.randint(0, self.graph.num_nodes() - 1), random.randint(0, self.graph.num_nodes() - 1)
            while self.graph.has_edges_between(src, dst) and (src, dst) not in neg and src != dst:
                src, dst = random.randint(0, self.graph.num_nodes() - 1), random.randint(0, self.graph.num_nodes() - 1)
            neg.append(((src, dst), False))

        # create raw dataset
        self.raw_dataset = pos + neg
        random.shuffle(self.raw_dataset)

    def sample_data_posteriors(self, train_val, val_test):
        """
        [1] Query target on two nodes
        [2] Get their posteriors for node classification
        [3] Concatinate posteriors to form the input feature vector for the attacker model
        """
        size = len(self.raw_dataset) - 1
        self.feature_amount = self.target_model.get_posteriors(self.modified_graph, 0).shape[0] * 2
        self.features = torch.zeros((size + 1, self.feature_amount), dtype=torch.float)
        self.labels = torch.zeros(size + 1, dtype=torch.long)

        for i, ((src, dst), label) in enumerate(self.raw_dataset):
            # query target model to get posteriors
            post_src = self.target_model.get_posteriors(self.modified_graph, src)
            post_dst = self.target_model.get_posteriors(self.modified_graph, dst)
            feature = torch.cat((post_src, post_dst))
            # create feature and label
            self.features[i] = feature
            self.labels[i] = torch.ones(1, 1) if label else torch.zeros(1, 1)

        # train, eval, test
        self.train_mask = torch.zeros(size, dtype=torch.bool)
        self.val_mask = torch.zeros(size, dtype=torch.bool)
        self.test_mask = torch.zeros(size, dtype=torch.bool)

        train_val_split = int(size * train_val)
        val_test_split  = train_val_split + int(size * val_test)

        # train masks
        for a in range(size):
            self.train_mask[a] = a < train_val_split
            self.val_mask[a] = a >= train_val_split and a < val_test_split
            self.test_mask[a] = a >= val_test_split

        # node ids
        self.train_nid = self.train_mask.nonzero().squeeze()
        self.val_nid = self.val_mask.nonzero().squeeze()
        self.test_nid = self.test_mask.nonzero().squeeze()

    def sample_data_vector_distances(self, train_val, val_test):
        """
        [1] Query target on two nodes
        [2] Get their posteriors for node classification
        [3] Create input feature vector for attacker model
            [3.1] Calculate distances between posterior vectors
            [3.2] Concatinate results to form a 8-dim input feature vector
        """
        size = len(self.raw_dataset) - 1
        self.feature_amount = 8
        self.features = torch.zeros((size + 1, self.feature_amount), dtype=torch.float)
        self.labels = torch.zeros(size + 1, dtype=torch.long)

        for i, ((src, dst), label) in enumerate(self.raw_dataset):
            # query target model to get posteriors
            post_src = self.target_model.get_posteriors(self.modified_graph, src)
            post_dst = self.target_model.get_posteriors(self.modified_graph, dst)
            src_list = [x.item() for x in post_src]
            dst_list = [x.item() for x in post_dst]

            # cosine distance
            # formular: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
            cosine_distance = distance.cosine(src_list, dst_list)

            # euclidean distance
            # formular: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html?highlight=euclidean#scipy.spatial.distance.euclidean
            euclidean_distance = distance.euclidean(src_list, dst_list)

            # correlation distance
            # formular: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html?highlight=correlation#scipy.spatial.distance.correlation
            correlation_distance = distance.correlation(src_list, dst_list)

            # chebyshev distance
            # formular https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.chebyshev.html?highlight=chebyshev#scipy.spatial.distance.chebyshev
            chebyshev_distance = distance.chebyshev(src_list, dst_list)

            # braycurtis distance
            # formular: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.braycurtis.html?highlight=braycurtis#scipy.spatial.distance.braycurtis
            braycurtis_distance = distance.braycurtis(src_list, dst_list)

            # manhattan distance
            # formular: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cityblock.html?highlight=manhattan
            manhattan_distance = distance.cityblock(src_list, dst_list)

            # canberra distance
            # formular: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.canberra.html?highlight=canberra#scipy.spatial.distance.canberra
            canberra_distance = distance.canberra(src_list, dst_list)

            # sqeuclidean distance
            # formular: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.sqeuclidean.html?highlight=sqeuclidean#scipy.spatial.distance.sqeuclidean
            sqeuclidean_distance = distance.sqeuclidean(src_list, dst_list)

            feature = torch.tensor([cosine_distance,
                                    euclidean_distance,
                                    correlation_distance,
                                    chebyshev_distance,
                                    braycurtis_distance,
                                    manhattan_distance,
                                    canberra_distance,
                                    sqeuclidean_distance])
            # create feature and label
            self.features[i] = feature
            self.labels[i] = torch.ones(1, 1) if label else torch.zeros(1, 1)

        # train, eval, test
        self.train_mask = torch.zeros(size, dtype=torch.bool)
        self.val_mask = torch.zeros(size, dtype=torch.bool)
        self.test_mask = torch.zeros(size, dtype=torch.bool)

        train_val_split = int(size * train_val)
        val_test_split  = train_val_split + int(size * val_test)

        # train masks
        for a in range(size):
            self.train_mask[a] = a < train_val_split
            self.val_mask[a] = a >= train_val_split and a < val_test_split
            self.test_mask[a] = a >= val_test_split

        # node ids
        self.train_nid = self.train_mask.nonzero().squeeze()
        self.val_nid = self.val_mask.nonzero().squeeze()
        self.test_nid = self.test_mask.nonzero().squeeze()


    def train(self, verbose=False):
        # initialize model
        self._initialize()

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

    def _initialize(self):
        # create model
        self.model = MLP(self.feature_amount,
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

    def evaluate(self, nid):
        self.model.eval()
        with torch.no_grad():
            # query model
            logits = self.model(self.features)
            logits = logits[nid]
            labels = self.labels[nid]
            _, pred = torch.max(logits, dim=1)

            # metrices
            precision = _precision(pred, labels)
            recall = _recall(pred, labels)
            f1_score = _f1_score(precision, recall)
            acc = _accuracy(pred, labels)

            return (precision, recall, f1_score, acc)
