# Python 3.8.5

import torch
import dgl
from dgl.data import citation_graph as citegrh
from dgl.data.minigc import *
from dgl.data.utils import *
from dgl.data.reddit import RedditDataset
import json


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

def error_msg(msg):
    print()
    print(f'  [-] Following Error occured: {msg}')
    print(f'  [-] Exiting..\n')
    exit(1)

def print_desc(opt, dir=''):
    with open(f'desc/{dir}{opt}.txt', 'r') as f:
        print(f.read()[:-1])

def print_datasets(d):
    possible_datasets = ['cora', 'citeseer', 'pubmed']
    if d != 'all' and d not in possible_datasets:
        error_msg(f'Unknown dataset \'{d}\'')
    datasets = possible_datasets if d == 'all' else [d]

    print(f'  [+] Datasets\n')
    print(f'       Name        Nodes     Edges     Features     Classes')
    print(f'      ------------------------------------------------------')
    for dataset in datasets:
        print_desc(dataset, 'datasets/')
    print('\n')

    return datasets

def print_gnns(gnn):
    possible_gnns = ['graphsage']
    if gnn != 'all' and gnn not in possible_gnns:
        error_msg(f'Unknown GNN \'{gnn}\'')
    gnns = possible_gnns if gnn == 'all' else [gnn]

    print(f'  [+] Graph Neural Networks\n')
    print(f'       Type         Aggregator Type')
    print(f'      ------------------------------')
    for gnn in gnns:
        with open(f'config/{gnn}.conf') as json_file:
            parameter = json.load(json_file)
            print(f'       {gnn}    {parameter["aggregator_type"]}')
    print('\n')

    return gnns
