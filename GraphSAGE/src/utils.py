# Python 3.8.5

import torch
import dgl
from dgl.data import citation_graph as citegrh
from dgl.data.minigc import *
from dgl.data.utils import *
from dgl.data.reddit import RedditDataset


class Dataset:

    def __init__(self, graph, features, labels, n_features, n_classes):

        self.graph = graph
        self.features = features
        self.labels = labels
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_nodes = self.graph.number_of_nodes()
        self.n_edges = self.graph.number_of_edges()

    def to(self, gpu):
        torch.cuda.set_device(gpu)
        self.features = self.features.cuda()
        self.labels = self.labels.cuda()
        self.train_mask = self.train_mask.cuda()
        self.val_mask = self.val_mask.cuda()
        self.test_mask = self.test_mask.cuda()

    def preprocess_data(self, cuda=-1):
        self.graph = dgl.remove_self_loop(self.graph)
        self.n_edges = self.graph.number_of_edges()
        if cuda > 0:
            self.graph = self.graph.int().to(cuda)

    def generate_masks(self, train, val, test):
        self.train_mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(self.n_nodes, dtype=torch.bool)

        train_val_split = int(self.n_nodes * train)
        val_test_split  = train_val_split + int(self.n_nodes * val)

        # masks
        for a in range(self.n_nodes):
            self.train_mask[a] = a < train_val_split
            self.val_mask[a] = a >= train_val_split and a < val_test_split
            self.test_mask[a] = a >= val_test_split

        # node ids
        self.train_nid = self.train_mask.nonzero().squeeze()
        self.val_nid = self.val_mask.nonzero().squeeze()
        self.test_nid = self.test_mask.nonzero().squeeze()

    def update(self, graph):
        self.graph = graph
        self.features = graph.ndata['feat']
        self.labels = graph.ndata['label']
        self.n_features = graph.ndata['feat'].shape[1]
        self.n_nodes = self.graph.number_of_nodes()
        self.n_edges = self.graph.number_of_edges()

def register_data_args(parser):
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="[cora, citeseer, pubmed, reddit]")

def load_data(dataset):
    if dataset == 'cora':
        return citegrh.load_cora(verbose=False)
    elif dataset == 'citeseer':
        return citegrh.load_citeseer(verbose=False)
    elif dataset == 'pubmed':
        return citegrh.load_pubmed(verbose=False)
    elif dataset is not None and dataset.startswith('reddit'):
        return RedditDataset(self_loop=('self-loop' in dataset))
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
