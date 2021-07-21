import matplotlib.pyplot as plt
import numpy
import pandas

data00 = pandas.read_csv('eval/005500-same-ds.txt')
data01 = pandas.read_csv('eval/050600-same-ds.txt')
data02 = pandas.read_csv('eval/081400-same-ds.txt')
data03 = pandas.read_csv('eval/112000-same-ds.txt')
data04 = pandas.read_csv('eval/173200-same-ds.txt')
data05 = pandas.read_csv('eval/224900-same-ds.txt')

datas0 = [data00, data01, data02, data03, data04, data05]


# -----------------------------------------------------------------------------------------------------------------------------
# Attack-1
# -----------------------------------------------------------------------------------------------------------------------------

attack_1 = {'graphsage': {}, 'gat': {}, 'gcn': {}}

attack_1['graphsage']['cora'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_1['graphsage']['citeseer'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_1['graphsage']['pubmed'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}

attack_1['gat']['cora'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_1['gat']['citeseer'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_1['gat']['pubmed'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}

attack_1['gcn']['cora'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_1['gcn']['citeseer'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_1['gcn']['pubmed'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}


# -----------------------------------------------------------------------------------------------------------------------------
# Attack-2
# -----------------------------------------------------------------------------------------------------------------------------

attack_2 = {'graphsage': {}, 'gat': {}, 'gcn': {}}

attack_2['graphsage']['cora'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_2['graphsage']['citeseer'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_2['graphsage']['pubmed'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}

attack_2['gat']['cora'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_2['gat']['citeseer'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_2['gat']['pubmed'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}

attack_2['gcn']['cora'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_2['gcn']['citeseer'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_2['gcn']['pubmed'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}

# Collect data
for data in datas0:

    for i, gnn in enumerate(data['GNN']):
        # Attack-1
        if 'baseline' in data['Attack'][i] and 'post' in data['Attack'][i]:
            attack_1[gnn][data['Dataset'][i]]['bl'].append(data['AF1'][i])

        elif '20p' in data['Attack'][i] and 'post' in data['Attack'][i]:
            attack_1[gnn][data['Dataset'][i]]['20'].append(data['AF1'][i])

        elif '40p' in data['Attack'][i] and 'post' in data['Attack'][i]:
            attack_1[gnn][data['Dataset'][i]]['40'].append(data['AF1'][i])

        elif '60p' in data['Attack'][i] and 'post' in data['Attack'][i]:
            attack_1[gnn][data['Dataset'][i]]['60'].append(data['AF1'][i])

        elif '80p' in data['Attack'][i] and 'post' in data['Attack'][i]:
            attack_1[gnn][data['Dataset'][i]]['80'].append(data['AF1'][i])  

        # Attack-2
        elif 'baseline' in data['Attack'][i] and 'dist' in data['Attack'][i]:
            attack_2[gnn][data['Dataset'][i]]['bl'].append(data['AF1'][i])

        elif '20p' in data['Attack'][i] and 'dist' in data['Attack'][i]:
            attack_2[gnn][data['Dataset'][i]]['20'].append(data['AF1'][i])

        elif '40p' in data['Attack'][i] and 'dist' in data['Attack'][i]:
            attack_2[gnn][data['Dataset'][i]]['40'].append(data['AF1'][i])

        elif '60p' in data['Attack'][i] and 'dist' in data['Attack'][i]:
            attack_2[gnn][data['Dataset'][i]]['60'].append(data['AF1'][i])

        elif '80p' in data['Attack'][i] and 'dist' in data['Attack'][i]:
            attack_2[gnn][data['Dataset'][i]]['80'].append(data['AF1'][i])      

gnns = ['graphsage', 'gat', 'gcn']
datasets = ['cora', 'citeseer', 'pubmed']
attacks = ['bl', '20', '40', '60', '80']

# Plot results
for dataset in datasets:
    # Attack 1
    plt_attack_1 = f'attack-1-{dataset}.png'
    res = {'graphsage': [], 'gat': [], 'gcn': []}

    for gnn in gnns:
        for attack in attacks:
            x = attack_1[gnn][dataset][attack]
            y = round(sum(x) / len(x), 4)
            res[gnn].append(y)
        
    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)
    
    bartypes = ['0', '20', '40', '60', '80']
    X_axis = numpy.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, res['graphsage'], size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(res['graphsage']):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, res['gat'], size, zorder = 3, label = 'GAT')
    for i, value in enumerate(res['gat']):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, res['gcn'], size, zorder = 3, label = 'GCN')
    for i, value in enumerate(res['gcn']):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')
    
    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.ylim([0, 1.1])
    plt.legend(loc = 'upper left')
    plt.savefig(plt_attack_1)

    # Attack 2
    plt_attack_2 = f'attack-2-{dataset}.png'
    res = {'graphsage': [], 'gat': [], 'gcn': []}

    for gnn in gnns:
        for attack in attacks:
            x = attack_2[gnn][dataset][attack]
            y = round(sum(x) / len(x), 4)
            res[gnn].append(y)
        
    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)
    
    bartypes = ['0', '20', '40', '60', '80']
    X_axis = numpy.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, res['graphsage'], size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(res['graphsage']):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, res['gat'], size, zorder = 3, label = 'GAT')
    for i, value in enumerate(res['gat']):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, res['gcn'], size, zorder = 3, label = 'GCN')
    for i, value in enumerate(res['gcn']):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')
    
    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.ylim([0, 1.1])
    plt.legend(loc = 'upper left')
    plt.savefig(plt_attack_2)

# -----------------------------------------------------------------------------------------------------------------------------
# Attack-3
# -----------------------------------------------------------------------------------------------------------------------------

data10 = pandas.read_csv('eval/104400-diff-ds.txt')
data11 = pandas.read_csv('eval/214400-diff-ds.txt')

datas1 = [data10, data11]

attack_3 = {'graphsage': {}, 'gat': {}, 'gcn': {}}

attack_3['graphsage']['cora_citeseer'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['graphsage']['cora_pubmed'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['graphsage']['citeseer_cora'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['graphsage']['citeseer_pubmed'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['graphsage']['pubmed_cora'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['graphsage']['pubmed_citeseer'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}

attack_3['gat']['cora_citeseer'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['gat']['cora_pubmed'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['gat']['citeseer_cora'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['gat']['citeseer_pubmed'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['gat']['pubmed_cora'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['gat']['pubmed_citeseer'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}

attack_3['gcn']['cora_citeseer'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['gcn']['cora_pubmed'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['gcn']['citeseer_cora'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['gcn']['citeseer_pubmed'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['gcn']['pubmed_cora'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}
attack_3['gcn']['pubmed_citeseer'] = {'bl': [], '20': [], '40': [], '60': [], '80': []}

# Collect data
for data in datas1:

    for i, gnn in enumerate(data['GNN']):
        # Attack-3
        for dataset in ['cora_citeseer', 'cora_pubmed', 'citeseer_cora', 'citeseer_pubmed', 'pubmed_cora', 'pubmed_citeseer']:
            if 'baseline' in data['Attack'][i] and dataset in data['Attack'][i]:
                attack_3[gnn][dataset]['bl'].append(data['AF1'][i])
            
            elif '20p' in data['Attack'][i] and dataset in data['Attack'][i]:
                attack_3[gnn][dataset]['20'].append(data['AF1'][i])
            
            elif '40p' in data['Attack'][i] and dataset in data['Attack'][i]:
                attack_3[gnn][dataset]['40'].append(data['AF1'][i])

            elif '60p' in data['Attack'][i] and dataset in data['Attack'][i]:
                attack_3[gnn][dataset]['60'].append(data['AF1'][i])

            elif '80p' in data['Attack'][i] and dataset in data['Attack'][i]:
                attack_3[gnn][dataset]['80'].append(data['AF1'][i])

gnns = ['graphsage', 'gat', 'gcn']
datasets = ['cora_citeseer', 'cora_pubmed', 'citeseer_cora', 'citeseer_pubmed', 'pubmed_cora', 'pubmed_citeseer']
attacks = ['bl', '20', '40', '60', '80']

# Plot results
for dataset in datasets:
    # Attack 3
    plt_attack_3 = f'attack-3-{dataset}.png'
    res = {'graphsage': [], 'gat': [], 'gcn': []}

    for gnn in gnns:
        for attack in attacks:
            x = attack_3[gnn][dataset][attack]
            y = round(sum(x) / len(x), 4)
            res[gnn].append(y)

    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)
    
    bartypes = ['0', '20', '40', '60', '80']
    X_axis = numpy.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, res['graphsage'], size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(res['graphsage']):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, res['gat'], size, zorder = 3, label = 'GAT')
    for i, value in enumerate(res['gat']):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, res['gcn'], size, zorder = 3, label = 'GCN')
    for i, value in enumerate(res['gcn']):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')
    
    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.ylim([0, 1.1])
    plt.legend(loc = 'upper left')
    plt.savefig(plt_attack_3)


# Statistics

# Attack 1
gnns = ['graphsage', 'gat', 'gcn']
datasets = ['cora', 'citeseer', 'pubmed']
attacks = ['bl', '20', '40', '60', '80']
# Avg. Baseline F1-Score
l = []
for gnn in gnns:
    for dataset in datasets:
        l.append(sum(attack_1[gnn][dataset]['bl']) / len(attack_1[gnn][dataset]['bl']))
avg_baseline_f1_score = round(sum(l) / len(l), 4)
# Best Baseline F1-Score
best_baseline_f1_score = (0, None, None)
for gnn in gnns:
    for dataset in datasets:
        if sum(attack_1[gnn][dataset]['bl']) / len(attack_1[gnn][dataset]['bl']) > best_baseline_f1_score[0]:
            best_baseline_f1_score = (sum(attack_1[gnn][dataset]['bl']) / len(attack_1[gnn][dataset]['bl']), gnn, dataset)
# Worst Baseline F1-Score
worst_baseline_f1_score = (1, None, None)
for gnn in gnns:
    for dataset in datasets:
        if sum(attack_1[gnn][dataset]['bl']) / len(attack_1[gnn][dataset]['bl']) < worst_baseline_f1_score[0]:
            worst_baseline_f1_score = (sum(attack_1[gnn][dataset]['bl']) / len(attack_1[gnn][dataset]['bl']), gnn, dataset)
# Avg. F1-Score of all Attack-1 Attacks
l = []
for gnn in gnns:
    for dataset in datasets:
        l.append(sum(attack_1[gnn][dataset]['bl']) / len(attack_1[gnn][dataset]['bl']))
        l.append(sum(attack_1[gnn][dataset]['20']) / len(attack_1[gnn][dataset]['20']))
        l.append(sum(attack_1[gnn][dataset]['40']) / len(attack_1[gnn][dataset]['40']))
        l.append(sum(attack_1[gnn][dataset]['60']) / len(attack_1[gnn][dataset]['60']))
        l.append(sum(attack_1[gnn][dataset]['80']) / len(attack_1[gnn][dataset]['80']))
        
avg_attack_1_f1_score = round(sum(l) / len(l), 4)
# Best F1-Score of all Attack-1 Attacks
best_attack_1_f1_score = (0, None, None, None)
for gnn in gnns:
    for dataset in datasets:
        for attack in ['bl', '20', '40','60', '80']:
            if sum(attack_1[gnn][dataset][attack]) / len(attack_1[gnn][dataset][attack]) > best_attack_1_f1_score[0]:
                best_attack_1_f1_score = (sum(attack_1[gnn][dataset][attack]) / len(attack_1[gnn][dataset][attack]), attack, gnn, dataset)
# Worst F1-Score of all Attack-1 Attacks
worst_attack_1_f1_score = (1, None, None, None)
for gnn in gnns:
    for dataset in datasets:
        for attack in ['bl', '20', '40','60', '80']:
            if sum(attack_1[gnn][dataset][attack]) / len(attack_1[gnn][dataset][attack]) < worst_attack_1_f1_score[0]:
                worst_attack_1_f1_score = (sum(attack_1[gnn][dataset][attack]) / len(attack_1[gnn][dataset][attack]), attack, gnn, dataset)

print()
print(f'Avg.  Attack-1 Baseline F1-Score: {avg_baseline_f1_score}')
print(f'Best  Attack-1 Baseline F1-Score: {best_baseline_f1_score}')
print(f'Worst Attack-1 Baseline F1-Score: {worst_baseline_f1_score}\n')
print(f'Avg.  Attack-1 F1-Score: {avg_attack_1_f1_score}')
print(f'Best  Attack-1 F1-Score: {best_attack_1_f1_score}')
print(f'Worst Attack-1 F1-Score: {worst_attack_1_f1_score}\n')

# Attack 2
# Avg. Baseline F1-Score
l = []
for gnn in gnns:
    for dataset in datasets:
        l.append(sum(attack_2[gnn][dataset]['bl']) / len(attack_2[gnn][dataset]['bl']))
avg_baseline_f1_score = round(sum(l) / len(l), 4)
# Best Baseline F1-Score
best_baseline_f1_score = (0, None, None)
for gnn in gnns:
    for dataset in datasets:
        if sum(attack_2[gnn][dataset]['bl']) / len(attack_2[gnn][dataset]['bl']) > best_baseline_f1_score[0]:
            best_baseline_f1_score = (sum(attack_2[gnn][dataset]['bl']) / len(attack_2[gnn][dataset]['bl']), gnn, dataset)
# Worst Baseline F1-Score
worst_baseline_f1_score = (1, None, None)
for gnn in gnns:
    for dataset in datasets:
        if sum(attack_2[gnn][dataset]['bl']) / len(attack_2[gnn][dataset]['bl']) < worst_baseline_f1_score[0]:
            worst_baseline_f1_score = (sum(attack_2[gnn][dataset]['bl']) / len(attack_2[gnn][dataset]['bl']), gnn, dataset)
# Avg. F1-Score of all Attack-2 Attacks
l = []
for gnn in gnns:
    for dataset in datasets:
        l.append(sum(attack_2[gnn][dataset]['bl']) / len(attack_2[gnn][dataset]['bl']))
        l.append(sum(attack_2[gnn][dataset]['20']) / len(attack_2[gnn][dataset]['20']))
        l.append(sum(attack_2[gnn][dataset]['40']) / len(attack_2[gnn][dataset]['40']))
        l.append(sum(attack_2[gnn][dataset]['60']) / len(attack_2[gnn][dataset]['60']))
        l.append(sum(attack_2[gnn][dataset]['80']) / len(attack_2[gnn][dataset]['80']))
        
avg_attack_2_f1_score = round(sum(l) / len(l), 4)
# Best F1-Score of all Attack-2 Attacks
best_attack_2_f1_score = (0, None, None, None)
for gnn in gnns:
    for dataset in datasets:
        for attack in ['bl', '20', '40','60', '80']:
            if sum(attack_2[gnn][dataset][attack]) / len(attack_2[gnn][dataset][attack]) > best_attack_2_f1_score[0]:
                best_attack_2_f1_score = (sum(attack_2[gnn][dataset][attack]) / len(attack_2[gnn][dataset][attack]), attack, gnn, dataset)
# Worst F1-Score of all Attack-2 Attacks
worst_attack_2_f1_score = (1, None, None, None)
for gnn in gnns:
    for dataset in datasets:
        for attack in ['bl', '20', '40','60', '80']:
            if sum(attack_2[gnn][dataset][attack]) / len(attack_2[gnn][dataset][attack]) < worst_attack_2_f1_score[0]:
                worst_attack_2_f1_score = (sum(attack_2[gnn][dataset][attack]) / len(attack_2[gnn][dataset][attack]), attack, gnn, dataset)

print()
print(f'Avg.  Attack-2 Baseline F1-Score: {avg_baseline_f1_score}')
print(f'Best  Attack-2 Baseline F1-Score: {best_baseline_f1_score}')
print(f'Worst Attack-2 Baseline F1-Score: {worst_baseline_f1_score}\n')
print(f'Avg.  Attack-2 F1-Score: {avg_attack_2_f1_score}')
print(f'Best  Attack-2 F1-Score: {best_attack_2_f1_score}')
print(f'Worst Attack-2 F1-Score: {worst_attack_2_f1_score}\n')


# Attack 3
datasets = ['cora_citeseer', 'cora_pubmed', 'citeseer_cora', 'citeseer_pubmed', 'pubmed_cora', 'pubmed_citeseer']
# Avg. Baseline F1-Score
l = []
for gnn in gnns:
    for dataset in datasets:
        l.append(sum(attack_3[gnn][dataset]['bl']) / len(attack_3[gnn][dataset]['bl']))
avg_baseline_f1_score = round(sum(l) / len(l), 4)
# Best Baseline F1-Score
best_baseline_f1_score = (0, None, None)
for gnn in gnns:
    for dataset in datasets:
        if sum(attack_3[gnn][dataset]['bl']) / len(attack_3[gnn][dataset]['bl']) > best_baseline_f1_score[0]:
            best_baseline_f1_score = (sum(attack_3[gnn][dataset]['bl']) / len(attack_3[gnn][dataset]['bl']), gnn, dataset)
# Worst Baseline F1-Score
worst_baseline_f1_score = (1, None, None)
for gnn in gnns:
    for dataset in datasets:
        if sum(attack_3[gnn][dataset]['bl']) / len(attack_3[gnn][dataset]['bl']) < worst_baseline_f1_score[0]:
            worst_baseline_f1_score = (sum(attack_3[gnn][dataset]['bl']) / len(attack_3[gnn][dataset]['bl']), gnn, dataset)
# Avg. F1-Score of all Attack-3 Attacks
l = []
for gnn in gnns:
    for dataset in datasets:
        l.append(sum(attack_3[gnn][dataset]['bl']) / len(attack_3[gnn][dataset]['bl']))
        l.append(sum(attack_3[gnn][dataset]['20']) / len(attack_3[gnn][dataset]['20']))
        l.append(sum(attack_3[gnn][dataset]['40']) / len(attack_3[gnn][dataset]['40']))
        l.append(sum(attack_3[gnn][dataset]['60']) / len(attack_3[gnn][dataset]['60']))
        l.append(sum(attack_3[gnn][dataset]['80']) / len(attack_3[gnn][dataset]['80']))
        
avg_attack_3_f1_score = round(sum(l) / len(l), 4)
# Best F1-Score of all Attack-3 Attacks
best_attack_3_f1_score = (0, None, None, None)
for gnn in gnns:
    for dataset in datasets:
        for attack in ['bl', '20', '40','60', '80']:
            if sum(attack_3[gnn][dataset][attack]) / len(attack_3[gnn][dataset][attack]) > best_attack_3_f1_score[0]:
                best_attack_3_f1_score = (sum(attack_3[gnn][dataset][attack]) / len(attack_3[gnn][dataset][attack]), attack, gnn, dataset)
# Worst F1-Score of all Attack-3 Attacks
worst_attack_3_f1_score = (1, None, None, None)
for gnn in gnns:
    for dataset in datasets:
        for attack in ['bl', '20', '40','60', '80']:
            if sum(attack_3[gnn][dataset][attack]) / len(attack_3[gnn][dataset][attack]) < worst_attack_3_f1_score[0]:
                worst_attack_3_f1_score = (sum(attack_3[gnn][dataset][attack]) / len(attack_3[gnn][dataset][attack]), attack, gnn, dataset)

print()
print(f'Avg.  Attack-3 Baseline F1-Score: {avg_baseline_f1_score}')
print(f'Best  Attack-3 Baseline F1-Score: {best_baseline_f1_score}')
print(f'Worst Attack-3 Baseline F1-Score: {worst_baseline_f1_score}\n')
print(f'Avg.  Attack-3 F1-Score: {avg_attack_3_f1_score}')
print(f'Best  Attack-3 F1-Score: {best_attack_3_f1_score}')
print(f'Worst Attack-3 F1-Score: {worst_attack_3_f1_score}\n')

# Best Attack Performance on GNNs
res = {'graphsage': [], 'gat': [], 'gcn': []}
# Attack 1
datasets = ['cora', 'citeseer', 'pubmed']
for gnn in gnns:
    best = 0
    for dataset in datasets:
        for attack in ['bl', '20', '40','60', '80']:
            if sum(attack_1[gnn][dataset][attack]) / len(attack_1[gnn][dataset][attack]) > best:
                best = sum(attack_1[gnn][dataset][attack]) / len(attack_1[gnn][dataset][attack])
    res[gnn].append(best)

# Attack 2
for gnn in gnns:
    best = 0
    for dataset in datasets:
        for attack in ['bl', '20', '40','60', '80']:
            if sum(attack_2[gnn][dataset][attack]) / len(attack_2[gnn][dataset][attack]) > best:
                best = sum(attack_2[gnn][dataset][attack]) / len(attack_2[gnn][dataset][attack])
    res[gnn].append(best)

# Attack 3
datasets = ['cora_citeseer', 'cora_pubmed', 'citeseer_cora', 'citeseer_pubmed', 'pubmed_cora', 'pubmed_citeseer']
for gnn in gnns:
    best = 0
    for dataset in datasets:
        for attack in ['bl', '20', '40','60', '80']:
            if sum(attack_3[gnn][dataset][attack]) / len(attack_3[gnn][dataset][attack]) > best:
                best = sum(attack_3[gnn][dataset][attack]) / len(attack_3[gnn][dataset][attack])
    res[gnn].append(best)


string = "Best Attack Performances LATEX Code:\n"
for gnn in gnns:
    string += f'{"GraphSAGE" if gnn == "graphsage" else gnn.upper()} & ${res[gnn][0]:0.4f}$ & ${res[gnn][1]:0.4f}$ & ${res[gnn][2]:0.4f}$ \\\\\n'

print(string)

# Average Attack Performance on GNNs
res = {'graphsage': [], 'gat': [], 'gcn': []}
# Attack 1
datasets = ['cora', 'citeseer', 'pubmed']
for gnn in gnns:
    l = []
    for dataset in datasets:
        l.append(sum(attack_1[gnn][dataset]['bl']) / len(attack_1[gnn][dataset]['bl']))
        l.append(sum(attack_1[gnn][dataset]['20']) / len(attack_1[gnn][dataset]['20']))
        l.append(sum(attack_1[gnn][dataset]['40']) / len(attack_1[gnn][dataset]['40']))
        l.append(sum(attack_1[gnn][dataset]['60']) / len(attack_1[gnn][dataset]['60']))
        l.append(sum(attack_1[gnn][dataset]['80']) / len(attack_1[gnn][dataset]['80']))
        
    res[gnn].append(round(sum(l) / len(l), 4))

# Attack 2
for gnn in gnns:
    l = []
    for dataset in datasets:
        l.append(sum(attack_2[gnn][dataset]['bl']) / len(attack_2[gnn][dataset]['bl']))
        l.append(sum(attack_2[gnn][dataset]['20']) / len(attack_2[gnn][dataset]['20']))
        l.append(sum(attack_2[gnn][dataset]['40']) / len(attack_2[gnn][dataset]['40']))
        l.append(sum(attack_2[gnn][dataset]['60']) / len(attack_2[gnn][dataset]['60']))
        l.append(sum(attack_2[gnn][dataset]['80']) / len(attack_2[gnn][dataset]['80']))
        
    res[gnn].append(round(sum(l) / len(l), 4))

# Attack 3
datasets = ['cora_citeseer', 'cora_pubmed', 'citeseer_cora', 'citeseer_pubmed', 'pubmed_cora', 'pubmed_citeseer']
for gnn in gnns:
    l = []
    for dataset in datasets:
        l.append(sum(attack_3[gnn][dataset]['bl']) / len(attack_3[gnn][dataset]['bl']))
        l.append(sum(attack_3[gnn][dataset]['20']) / len(attack_3[gnn][dataset]['20']))
        l.append(sum(attack_3[gnn][dataset]['40']) / len(attack_3[gnn][dataset]['40']))
        l.append(sum(attack_3[gnn][dataset]['60']) / len(attack_3[gnn][dataset]['60']))
        l.append(sum(attack_3[gnn][dataset]['80']) / len(attack_3[gnn][dataset]['80']))
        
    res[gnn].append(round(sum(l) / len(l), 4))

string = "Avg. Attack Performances LATEX Code:\n"
for gnn in gnns:
    string += f'{"GraphSAGE" if gnn == "graphsage" else gnn.upper()} & ${res[gnn][0]:0.4f}$ & ${res[gnn][1]:0.4f}$ & ${res[gnn][2]:0.4f}$ \\\\\n'

print(string)