from util import DataParser
import ast
import numpy as np
import joblib as jl


c=0
x_dataset = []
y_dataset = []
for f in ['log1.txt', 'log2.txt', 'log3.txt']:#
    with open(f, 'r') as fr:
        for line in fr:
            if('TBL_TRANS' in line):
                try:
                    transactions_str = line.replace('\n', '').split('TBL_TRANS=')[1]
                    transactions = ast.literal_eval(transactions_str)
                    print('begin round-{0}, trans_len={1}'.format(c, len(transactions)))
                    
                    dp = DataParser(transactions)
                    x_data,y_data = dp.get_dataset()
                    x_dataset += x_data
                    y_dataset += y_data
                    c+=1
                    
                except Exception as e:
                    print('exception:{0}'.format(e))
                
x_dataset = np.array(x_dataset)
y_dataset = np.array(y_dataset)
y_dataset_tmp = []
for y in y_dataset:
    if(y>0):
        y_dataset_tmp.append(1)
    else:
        y_dataset_tmp.append(0)
y_dataset = np.array(y_dataset_tmp)


print('x_dataset.shape={0}'.format(x_dataset.shape))
print('y_dataset.shape={0}'.format(y_dataset.shape))

jl.dump(x_dataset, 'x_dataset.jl')
jl.dump(y_dataset, 'y_dataset.jl')




