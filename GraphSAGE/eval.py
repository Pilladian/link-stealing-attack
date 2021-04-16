# Python 3.8.5

import argparse
import inductive

def main(args):

    tacc, aacc = [], []
    print(f'\n [+] Inductive setting')
    print(f' [+] Initialize attacks for dataset {args.dataset}')
    for i in range(args.iter):
        print(f' [+] Attack {i + 1} of {args.iter} running...', end='\r')
        t, a = inductive.main(args.dataset)
        tacc.append(t)
        aacc.append(a)

    print(f'\n [+] Done')
    print(f'\n [  Target  ] {round(sum(tacc) / len(tacc), 4):.4f}')
    print(f' [ Attacker ] {round(sum(aacc) / len(aacc), 4):.4f}\n')


if __name__ == '__main__':

    # cmd args
    parser = argparse.ArgumentParser(description='Link-Stealing Attack')
    parser.add_argument("--dataset",
                        type=str,
                        default="cora",
                        required=False,
                        help="[cora, citeseer, pubmed, reddit]")
    parser.add_argument("--iter",
                        type=int,
                        default=10,
                        required=False,
                        help="Amount of iterations")

    parser.add_argument("--ind",
                        action='store_true',
                        help="Set flag for inductive attacks")

    args = parser.parse_args()
    main(args)
