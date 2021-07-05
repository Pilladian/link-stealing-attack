# Python 3.8.5

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Imports
import argparse
from src.utils import print_init, print_datasets, print_gnns, final_evaluation_same_domain, eval_one_attack
from src import Experiment


# Attack 1 & 2 : Same Distribution Attacks
def main(args):
    # Init
    print_init(args)

    # Datasets
    datasets = print_datasets(args.dataset)
    # Graph Neural Networks
    gnns = print_gnns(args.gnn)

    # create Experiments
    experiments = []
    for gnn in gnns:
        for dataset in datasets:
            exp = Experiment(gnn, dataset, args.verbose)
            exp.initialize()
            experiments.append(exp)

    # run attacks
    for experiment in experiments:
        print(f'\n\n  [+] Same Domain - Posteriors - {experiment.gnn_name} : {experiment.dataset_name}\n')
        # Attack 1 : concatenation of posteriors as features
        #experiment.baseline_train_same_domain_post()
        #experiment.baseline_test_same_domain_post()
        #experiment.surviving_edges_same_domain_post(0.20)
        #experiment.surviving_edges_same_domain_post(0.40)
        experiment.surviving_edges_same_domain_post(0.60)
        #experiment.surviving_edges_same_domain_post(0.80)

        print(f'\n\n  [+] Same Domain - Distances - {experiment.gnn_name} : {experiment.dataset_name}\n')
        # Attack 2 : concatenation of 8 distance values as features
        #experiment.baseline_train_same_domain_dist()
        #experiment.baseline_test_same_domain_dist()
        #experiment.surviving_edges_same_domain_dist(0.20)
        #experiment.surviving_edges_same_domain_dist(0.40)
        #experiment.surviving_edges_same_domain_dist(0.60)
        #experiment.surviving_edges_same_domain_dist(0.80)

    #final_evaluation_same_domain(experiments, log=args.log, clear=args.clear)
    gnn = "gcn"
    dataset = "pubmed"
    attack = "surviving_edges_same_domain_post_60p" 
    f = f"results/{gnn}/{dataset}/{attack}.csv" 

    eval_one_attack(experiments, f)


if __name__ == '__main__':

    # cmd args
    parser = argparse.ArgumentParser(description='Link-Stealing Attack')

    parser.add_argument("--log",
                        action="store_true",
                        help="Log Output in ./log/lineup.txt and ./log/results.json")

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
