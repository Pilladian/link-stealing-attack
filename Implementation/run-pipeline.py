# Python 3.8.5

import argparse

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

    print(f'  [+] Following datasets have been selected\n')
    print(f'       Name        Nodes     Edges     Features     Classes')
    print(f'      ------------------------------------------------------')
    for dataset in datasets:
        print_desc(dataset, 'datasets/')
    print()


def main(args):
    # [0] Init
    print_desc('init')

    # [1] Print Datasets
    print_datasets(args.dataset)

    # [2] Create Baseline ( only node features )
        # [2.0] [Toggle] Description
        # [2.1] Target Model
            # [2.1.1] Parameter
            # [2.1.2] Dataset-Stats
            # [2.1.3] Parameter

        # [2.2] Attacker Model
            # [2.2.1] Parameter
            # [2.2.2] Dataset-Stats
            # [2.2.3] Parameter


    # [3] 1. Modification ( more Edges )

    # [4] 2. Modification ( different  )

if __name__ == '__main__':

    # cmd args
    parser = argparse.ArgumentParser(description='Link-Stealing Attack')
    parser.add_argument("--desc",
                        action="store_false",
                        help="Ouput Process Descriptions")

    parser.add_argument("--dataset",
                        type=str,
                        default="all",
                        help="Select Dataset [all, cora, citeseer, pubmed]")

    parser.add_argument("--gnn",
                        type=str,
                        default='all',
                        help="Type of GNN [all, graphsage, ...]")


    args = parser.parse_args()
    main(args)
