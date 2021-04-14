# Python 3.8.5

import dgl
from dgl.data import citation_graph as citegrh
from dgl.data.minigc import *
from dgl.data.utils import *
from dgl.data.reddit import RedditDataset


class Dataset:

    def __init__(self, graph, features, labels, train_mask, val_mask,
                 test_mask, n_features, n_classes, n_edges):

        self.graph = graph
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.train_nid = train_mask.nonzero().squeeze()
        self.val_nid = val_mask.nonzero().squeeze()
        self.test_nid = test_mask.nonzero().squeeze()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_edges = n_edges

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


def register_data_args(parser):
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="[cora, citeseer, pubmed, syn(synthetic dataset), reddit]")

def load_data(args):
    if args.dataset == 'cora':
        return citegrh.load_cora(verbose=False)
    elif args.dataset == 'citeseer':
        return citegrh.load_citeseer(verbose=False)
    elif args.dataset == 'pubmed':
        return citegrh.load_pubmed(verbose=False)
    elif args.dataset is not None and args.dataset.startswith('reddit'):
        return RedditDataset(self_loop=('self-loop' in args.dataset))
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
