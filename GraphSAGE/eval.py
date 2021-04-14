# Python 3.8.5

import argparse
import attack

def main(args):

    tacc, aacc = [], []
    print(f'\n [+] Initialize Attacks for dataset {args.dataset}')
    for i in range(args.iter):
        print(f' [+] Attack {i + 1} of {args.iter} running...', end='\r')
        t, a = attack.main(args.dataset)
        tacc.append(t)
        aacc.append(a)

    print(f'\n [+] Done')
    print(f'\n [ Avg. Acc. Target ] {round(sum(tacc) / len(tacc), 4)}')
    print(f' [ Avg. Acc. Attack ] {round(sum(aacc) / len(aacc), 4)}\n')


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

    args = parser.parse_args()
    main(args)
