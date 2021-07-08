import pandas

data1 = pandas.read_csv('diff-domain/202107060628-diff-ds-lineup.txt')
data2 = pandas.read_csv('diff-domain/202107062225-diff-ds-lineup.txt')
data3 = pandas.read_csv('diff-domain/202107071946-diff-ds-lineup.txt')

datas = [data1, data2, data3]


for dsets in ['cora_citeseer', 'cora_pubmed', 'citeseer_cora', 'citeseer_pubmed', 'pubmed_cora', 'pubmed_citeseer']:
    
    text = f"{dsets}\n"
    
    for sgnn in ['graphsage', 'gat', 'gcn']:

        train = []
        p20 = []
        p40 = []
        p60 = []
        p80 = []

        for data in datas:
            for i, gnn in enumerate(data['GNN']):

                if sgnn in gnn:
                    if dsets in data['Attack'][i]:
                        if 'train' in data['Attack'][i]:
                            train.append(data['AF1'][i])
                        if '20p' in data['Attack'][i]:
                            p20.append(data['AF1'][i])
                        if '40p' in data['Attack'][i]:
                            p40.append(data['AF1'][i])
                        if '60p' in data['Attack'][i]:
                            p60.append(data['AF1'][i])
                        if '80p' in data['Attack'][i]:
                            p80.append(data['AF1'][i])

        trains = f'{sum(train) / len(train):0.4f}'
        p20s = f'{sum(p20) / len(p20):0.4f}'
        p40s = f'{sum(p40) / len(p40):0.4f}'
        p60s = f'{sum(p60) / len(p60):0.4f}'
        p80s = f'{sum(p80) / len(p80):0.4f}'

        result = [trains, p20s, p40s, p60s, p80s]
        
        text += f"{sgnn} = [{trains}, {p20s}, {p40s}, {p60s}, {p80s}]\n"
    print(text + '\n')