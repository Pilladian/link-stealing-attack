# Python 3.8.5

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Imports
import argparse
import random
import os
import dgl
import torch
from src.utils import load_data, print_datasets, print_gnns
from src import Target
from dgl.data.utils import save_graphs


class Trainer:

    def __init__(self, ds_name, gnn_name):
        self.dataset_name = ds_name
        self.gnn_name = gnn_name

        # load dataset
        self.dataset = load_data(self.dataset_name)
        self.original_graph = self.dataset[0]
        self.num_classes = self.dataset.num_classes

    def _save_model(self, acc):
        # create models directory if not already exists
        dir = './models/'
        if not os.path.isdir(dir):
            os.system(f'mkdir {dir}')

        dir += f'{self.dataset_name}/'
        if not os.path.isdir(dir):
            os.system(f'mkdir {dir}')

        dir += f'{self.gnn_name}/'
        if not os.path.isdir(dir):
            os.system(f'mkdir {dir}')

        # save current state of model
        torch.save(self.target.model.state_dict(), f'{dir}model.pt')
        # save train and test graph
        save_graphs(f'{dir}traingraph.bin', [self.traingraph], {"label": self.traingraph.ndata['label']})
        save_graphs(f'{dir}testgraph.bin', [self.testgraph], {"label": self.testgraph.ndata['label']})
        # save acc
        os.system(f'echo {acc} > {dir}acc.txt')

    def _get_best_model(self, n):
        print()
        best = self._load_best_acc()
        for i in range(n):
            self._split_dataset()
            self._create_target_model()
            acc = self.target.evaluate(self.testgraph)
            if acc > best:
                best = acc
                print(f'  [+] {self.dataset_name}-{self.gnn_name}: {acc}', end='\r')
                self._save_model(acc)

    def _load_best_acc(self):
        dir = './models/'
        if not os.path.isdir(dir):
            os.system(f'mkdir {dir}')

        dir += f'{self.dataset_name}/'
        if not os.path.isdir(dir):
            os.system(f'mkdir {dir}')

        dir += f'{self.gnn_name}/'
        if not os.path.isdir(dir):
            os.system(f'mkdir {dir}')

        try:
            with open(f'{dir}acc.txt', 'r') as t:
                l = t.readlines()
                return float(l[0][:-1])
        except:
            return 0.0

    def _split_dataset(self):
        train_mask = torch.zeros(self.original_graph.number_of_nodes(), dtype=torch.bool)
        test_mask = torch.zeros(self.original_graph.number_of_nodes(), dtype=torch.bool)

        for a in range(self.original_graph.number_of_nodes()):
            val = random.choice([True, False])
            train_mask[a] = val
            test_mask[a] = not val

        traingraph = self.original_graph.subgraph(train_mask)
        testgraph = self.original_graph.subgraph(test_mask)

        # remove self loops
        self.traingraph = dgl.remove_self_loop(traingraph)
        self.testgraph = dgl.remove_self_loop(testgraph)

    def _create_target_model(self):
        self.target = Target(self.gnn_name, self.traingraph, self.num_classes)
        self.target.train()


def main(args):

    # dataset
    datasets = print_datasets(args.dataset)
    # GNNs
    gnns = print_gnns(args.gnn)

    for dataset in datasets:
        for gnn in gnns:
            trainer = Trainer(dataset, gnn)
            trainer._get_best_model(50)
    print()


if __name__ == '__main__':

    # cmd args
    parser = argparse.ArgumentParser(description='Link-Stealing Attack')

    parser.add_argument("--dataset",
                        type=str,
                        default="all",
                        help="Select Dataset [all, cora, citeseer, pubmed]")

    parser.add_argument("--gnn",
                        type=str,
                        default='all',
                        help="Type of GNN [all, graphsage, gcn, gat]")

    args = parser.parse_args()
    main(args)
