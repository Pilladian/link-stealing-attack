# Python 3.8.5

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Imports
import argparse
from src.utils import print_init, print_datasets, print_gnns, final_evaluation_diff_domain_2
from src import Experiment


# Attack 3 : Different Distribution Attacks
def main(args):
    # Init
    print_init(args)

    # Datasets
    datasets = print_datasets(args.dataset)
    # GNNs
    gnns = print_gnns(args.gnn)

    # target trained on different domain than attacker uses for queries
    diff_domain_experiments = []
    if len(datasets) > 1:
        for gnn in gnns:
            for d1 in datasets:
                for d2 in datasets:
                    if d1 != d2:
                        print(f'\n\n  [+] Different Domain - Distances - {gnn} : {d1} : {d2}\n')
                        exp = Experiment(gnn, d1, args.verbose)
                        exp.initialize()
                        exp.baseline_train_diff_domain_dist(d1, d2)
                        #exp.baseline_test_diff_domain_dist(d1, d2)
                        exp.surviving_edges_diff_domain_dist(0.20, d1, d2)
                        exp.surviving_edges_diff_domain_dist(0.40, d1, d2)
                        exp.surviving_edges_diff_domain_dist(0.60, d1, d2)
                        exp.surviving_edges_diff_domain_dist(0.80, d1, d2)
                        diff_domain_experiments.append(exp)
        final_evaluation_diff_domain_2(diff_domain_experiments)


if __name__ == '__main__':

    # cmd args
    parser = argparse.ArgumentParser(description='Link-Stealing Attack')

    parser.add_argument("--log",
                        action="store_true",
                        help="Log Ouput in ./log/lineup.txt and ./log/results.json")

    parser.add_argument("--clear",
                        action="store_true",
                        help="Clear all files in ./log/")

    parser.add_argument("--verbose",
                        action="store_true",
                        help="Output Attack results")

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
