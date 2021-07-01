import os
import pandas

gnn = "graphsage"
dataset = "pubmed"
attack = "surviving_edges_same_domain_dist_40p"

f = f"results/{gnn}/{dataset}/{attack}.csv"
with open(f, 'a') as fff:
        fff.write(f"acc,f1\n")

for a in range(10):
    os.system(f"python run-sd-attacks.py --gnn {gnn} --dataset {dataset}")

data = pandas.read_csv(f)

avg_acc = sum(data['acc']) / len(data['acc'])
avg_f1 = sum(data['f1']) / len(data['f1'])

with open(f"results/{gnn}/{dataset}/results.csv", 'a') as fff:
    fff.write(f"{attack},{avg_acc},{avg_f1}\n")