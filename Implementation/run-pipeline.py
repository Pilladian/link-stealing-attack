# Python 3.8.5

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Imports
import argparse
import json
import os
from src.utils import *
from src import *


class Attack:

    def __init__(self, gnn, dataset):
        self.gnn_type = gnn
        self.dataset_name = dataset

    def initialize(self):
        # load dataset
        self.dataset = load_data(self.dataset_name)
        self.original_graph = self.dataset[0]
        self.num_classes = self.dataset.num_classes
        # split_dataset()
        self._split_dataset()

    def _split_dataset(self):
        split = self.original_graph.number_of_nodes() * 0.5
        train_mask = torch.zeros(self.original_graph.number_of_nodes(), dtype=torch.bool)
        test_mask = torch.zeros(self.original_graph.number_of_nodes(), dtype=torch.bool)

        for a in range(self.original_graph.number_of_nodes()):
            train_mask[a] = a < split
            test_mask[a] = a >= split

        traingraph = self.original_graph.subgraph(train_mask)
        testgraph = self.original_graph.subgraph(test_mask)

        # remove self loops
        self.traingraph = dgl.remove_self_loop(traingraph)
        self.testgraph = dgl.remove_self_loop(testgraph)

    def create_baseline(self):
        # Baseline 1 - Train on traingraph - Test on traingraph
        # target
        self.bl1_target = Target(self.gnn_type, self.traingraph, self.num_classes)
        self.bl1_target.train()

        # attacker
        self.bl1_attacker = Attacker(self.bl1_target, self.traingraph)
        self.bl1_attacker.create_modified_graph(0)   # remove all edges ( 0% ) -> only node features
        self.bl1_attacker.sample_data(0.2, 0.4)
        self.bl1_attacker.train()

        # Baseline 2 - Train on traingraph - Test on testgraph
        # target
        self.bl2_target = Target(self.gnn_type, self.traingraph, self.num_classes)
        self.bl2_target.train()

        # attacker
        self.bl2_attacker = Attacker(self.bl1_target, self.testgraph)
        self.bl2_attacker.create_modified_graph(0)   # remove all edges ( 0% ) -> only node features
        self.bl2_attacker.sample_data(0.2, 0.4)
        self.bl2_attacker.train()

        # Evaluation
        self.bl1_target_result = self.bl1_target.evaluate(self.testgraph)
        self.bl1_attacker_result = self.bl1_attacker.evaluate(self.bl1_attacker.test_nid)
        self.bl2_target_result = self.bl2_target.evaluate(self.testgraph)
        self.bl2_attacker_result = self.bl2_attacker.evaluate(self.bl2_attacker.test_nid)



def main(args):
    # Init
    os.system('clear')
    print_desc('init')
    # Datasets
    datasets = print_datasets(args.dataset)
    # GNNs
    gnns = print_gnns(args.gnn)

    # create Attack objects
    attacks = []
    for gnn in gnns:
        for dataset in datasets:
            attack = Attack(gnn, dataset)
            attack.initialize()
            attacks.append(attack)

    # [2] Create Baseline ( only node features )
    print('  [+] Train Baseline Models\n')
    print('       GNN         Dataset     B1 Target     B1 Attacker     B2 Target     B2 Attacker')
    print(f'      ---------------------------------------------------------------------------------')
    for i, attack in enumerate(attacks):
        attack.create_baseline()
        print(f'       {attack.gnn_type}   '
              f'{attack.dataset_name}{" " * (12 - len(attack.dataset_name))}'
              f'{attack.bl1_target_result:0.4f}        '
              f'{attack.bl1_attacker_result:0.4f}          '
              f'{attack.bl2_target_result:0.4f}        '
              f'{attack.bl2_attacker_result:0.4f}         ')
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


#       GNN         Dataset     B1 Target     B1 Attacker     B2 Target     B2 Attacker
#      ---------------------------------------------------------------------------------
#       graphsage   cora        0.7851        0.7411          0.8006        0.6802
#       graphsage   citeseer    0.7264        0.7214          0.7222        0.7206
#       graphsage   pubmed      0.8497        0.6535          0.8513        0.6875

# add time
