# Python 3.8.5

import torch
import dgl
from dgl.data import citation_graph as citegrh
from dgl.data.minigc import *
from dgl.data.utils import *
from dgl.data.reddit import RedditDataset
from dgl.data.ppi import PPIDataset
import json
import os
from datetime import datetime


def register_data_args(parser):
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="[cora, citeseer, pubmed, ppi]")

def load_data(dataset):
    if dataset == 'cora':
        return citegrh.load_cora(verbose=False)
    elif dataset == 'citeseer':
        return citegrh.load_citeseer(verbose=False)
    elif dataset == 'pubmed':
        return citegrh.load_pubmed(verbose=False)
    elif dataset == 'ppi':
        return PPIDataset()
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

def print_init(args):
    os.system('clear')
    print_desc('init')
    print(f'  [+] Verbose Output {"enabled" if args.log else "disabled"}')
    print(f'  [+] Logging {"enabled" if args.log else "disabled"}\n\n')

def print_datasets(d):
    d = [s.strip() for s in d.split(",")]
    possible_datasets = ['cora', 'citeseer', 'pubmed', 'ppi']
    for ds in d:
        if ds != 'all' and ds not in possible_datasets:
            error_msg(f'Unknown dataset \'{ds}\'')

    datasets = possible_datasets if 'all' in d else d

    print(f'  [+] Datasets\n')
    print(f'      Name        Nodes     Edges     Features     Classes')
    print(f'      ----------------------------------------------------')
    for dataset in datasets:
        print_desc(dataset, 'datasets/')
    print('\n')

    return datasets

def print_gnns(gnns):
    gnns = [s.strip() for s in gnns.split(",")]
    gnn_types = ['graphsage', 'gat', 'gcn']
    for g in gnns:
        if g != 'all' and g not in gnn_types:
            error_msg(f'Unknown GNN type \'{g}\'')

    gnns = gnn_types if 'all' in gnns else gnns

    print(f'  [+] Graph Neural Networks\n')
    print(f'      Type')
    print(f'      ----------')
    for gnn in gnns:
        print(f'      {gnn}')

    return gnns

def print_attack_start(name):
    print(f'                                                              ', end='\r')
    print(f'      [-] {name} - currently running...', end='\r')

def print_attack_done(name):
    print(f'                                                              ', end='\r')
    print(f'      [-] {name} - done..', end='\r')
    print()

def print_attack_results(tacc, aprec, arecall, af1, aacc):
    print(f'''\n          Metric       Target      Attacker
          ---------------------------------
          Precision      -         {aprec:.4f}
          Recall         -         {arecall:.4f}
          F1-Score       -         {af1:.4f}
          Accuracy     {tacc:.4f}      {aacc:.4f}\n''')

def final_evaluation(experiments, log, clear):
    if clear:
        os.system('rm ./log/*')
    lineup_file = f'./log/lineup-{datetime.now().strftime("%Y-%m-%d-%H-%M")}.txt'
    with open(lineup_file, 'w') as lineup:

        lineup.write('''Attack                    GNN           Dataset       Target Acc      Attacker Acc      Attacker F1-Score\n''')
        lineup.write('''---------------------------------------------------------------------------------------------------------\n''')

        for i, exp in enumerate(experiments):
            for a in list(experiments[0].results.keys()):
                lineup.write(f'{a}{" " * (26 - len(a))}{exp.gnn_name}{" " * (14 - len(exp.gnn_name))}{exp.dataset_name}{" " * (14 - len(exp.dataset_name))}{exp.results[a]["target"]["acc"]*100:.2f}{" " * 11}{exp.results[a]["attacker"]["acc"]*100:.2f}{" " * 13}{exp.results[a]["attacker"]["f1-score"]*100:.2f}\n')
            lineup.write('\n')

    if log:
        res = dict()
        for ind, exp in enumerate(experiments):
            gnn = exp.gnn_name
            if gnn not in res:
                res[gnn] = {}

            ds = exp.dataset_name
            if ds not in res[gnn]:
                res[gnn][ds] = {}

            for attack, vals in exp.results.items():
                res[gnn][ds][attack] = {}
                for obj, metrics in vals.items():
                    res[gnn][ds][attack][obj] = {}
                    for metric, value in metrics.items():
                        res[gnn][ds][attack][obj][metric] = value.item()

        with open('./log/results.json', 'w') as jf:
            json_string = json.dumps(res, indent = 4)
            jf.write(json_string)

    print('\n')
    print(f'  [+] Lineup of all Attacks - cat from {lineup_file}\n')
    with open(lineup_file, 'r') as out:
        lines = out.readlines()
        string = ""
        for line in lines:
            string += '      ' + line
        print(string)
