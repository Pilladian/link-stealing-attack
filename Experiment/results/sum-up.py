import pandas

data1 = pandas.read_csv('diff-domain/202107060628-diff-ds-lineup.txt')
data2 = pandas.read_csv('diff-domain/202107062225-diff-ds-lineup.txt')
data3 = pandas.read_csv('diff-domain/202107071946-diff-ds-lineup.txt')

datas = [data1, data2, data3]

attacks = []
attack = 'all'
maximum = (0, None, None)
minimum = (1, None, None)
improvement = (0, None, None)


# for dsets in ['cora_citeseer', 'cora_pubmed', 'citeseer_cora', 'citeseer_pubmed', 'pubmed_cora', 'pubmed_citeseer']:
    
#     text = f"{dsets}\n"
    
#     for sgnn in ['graphsage', 'gat', 'gcn']:

#         train = []
#         p20 = []
#         p40 = []
#         p60 = []
#         p80 = []

#         for data in datas:
#             for i, gnn in enumerate(data['GNN']):

#                 if sgnn in gnn:
#                     if dsets in data['Attack'][i]:
#                         if 'train' in data['Attack'][i]:
#                             train.append(data['AF1'][i])
#                         if '20p' in data['Attack'][i]:
#                             p20.append(data['AF1'][i])
#                         if '40p' in data['Attack'][i]:
#                             p40.append(data['AF1'][i])
#                         if '60p' in data['Attack'][i]:
#                             p60.append(data['AF1'][i])
#                         if '80p' in data['Attack'][i]:
#                             p80.append(data['AF1'][i])

#         trains = f'{sum(train) / len(train):0.4f}'
#         p20s = f'{sum(p20) / len(p20):0.4f}'
#         p40s = f'{sum(p40) / len(p40):0.4f}'
#         p60s = f'{sum(p60) / len(p60):0.4f}'
#         p80s = f'{sum(p80) / len(p80):0.4f}'

#         if attack == 'baseline':
#             attacks.append(float(trains))
#         elif attack == '20p':
#             attacks.append(float(p20s))
#         elif attack == '40p':
#             attacks.append(float(p40s))
#         elif attack == '60p':
#             attacks.append(float(p60s))
#         elif attack == '80p':
#             attacks.append(float(p80s))
#         elif attack == 'all':
#             attacks.append(float(trains))
#             attacks.append(float(p20s))
#             attacks.append(float(p40s))
#             attacks.append(float(p60s))
#             attacks.append(float(p80s))
        
#         for res in [float(trains), float(p20s), float(p40s), float(p60s), float(p80s)]:
#             if res > maximum[0]:
#                 maximum = (res, dsets, sgnn)
            
#             if res < minimum[0]:
#                 minimum = (res, dsets, sgnn)

#             impr = max([float(p20s), float(p40s), float(p60s), float(p80s)]) - float(trains)
#             if impr > improvement[0]:
#                 improvement = (impr, dsets, sgnn)
            
        
#         text += f"{dsets}_{sgnn} = [{trains}, {p20s}, {p40s}, {p60s}, {p80s}]\n"
#     print(text + '\n')

for sgnn in ['graphsage', 'gat', 'gcn']:
    
    for dsets in ['cora_citeseer', 'cora_pubmed', 'citeseer_cora', 'citeseer_pubmed', 'pubmed_cora', 'pubmed_citeseer']:

        text = f"{dsets}\n"
        
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

        if attack == 'baseline':
            attacks.append(float(trains))
        elif attack == '20p':
            attacks.append(float(p20s))
        elif attack == '40p':
            attacks.append(float(p40s))
        elif attack == '60p':
            attacks.append(float(p60s))
        elif attack == '80p':
            attacks.append(float(p80s))
        elif attack == 'all':
            attacks.append(float(trains))
            attacks.append(float(p20s))
            attacks.append(float(p40s))
            attacks.append(float(p60s))
            attacks.append(float(p80s))
        
        for res in [float(trains), float(p20s), float(p40s), float(p60s), float(p80s)]:
            if res > maximum[0]:
                maximum = (res, dsets, sgnn)
            
            if res < minimum[0]:
                minimum = (res, dsets, sgnn)

            impr = max([float(p20s), float(p40s), float(p60s), float(p80s)]) - float(trains)
            if impr > improvement[0]:
                improvement = (impr, dsets, sgnn)
            
        
        text += f"{dsets}_{sgnn} = [{trains}, {p20s}, {p40s}, {p60s}, {p80s}]\n"
    #print(text + '\n')
    print(f'Avg. F1-Score {sgnn} - {attack} {sum(attacks) / len(attacks):.4f}')
    attacks = []

#print(f'Avg. F1-Score {attack} {sum(attacks) / len(attacks):.4f}')
#print(f'Maximum F1-Score: {maximum[0]:.4f} - {maximum[1]} - {maximum[2]}')
#print(f'Minimum F1-Score: {minimum[0]:.4f} - {minimum[1]} - {minimum[2]}')
#print(f'Improvement: {improvement[0]:.4f} - {improvement[1]} - {improvement[2]}')
