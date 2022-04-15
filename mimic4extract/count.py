import pandas as pd 
import numpy as np


def count_class(values, class_label):
    return np.where(values==class_label)[0].shape[0]

def get_data_stats(data_root, dataset='mimic-iii'):


    train = pd.read_csv(f'{data_root}/train_listfile.csv').y_true.values
    val = pd.read_csv(f'{data_root}/val_listfile.csv').y_true.values
    test = pd.read_csv(f'{data_root}/test_listfile.csv').y_true.values
    total_0 = 0
    total_1 = 0
    total_0 = count_class(train, 0) + count_class(val, 0) + count_class(test, 0)
    total_1 = count_class(train, 1) + count_class(val, 1) + count_class(test, 1)
    print(f'{dataset}')
    print(f'overall 0s {total_0}  1s {total_1}')
    print(f'train  0s {count_class(train, 0)}  1s {count_class(train, 1)}')
    print(f'val  0s {count_class(val, 0)}  1s {count_class(val, 1)}')
    print(f'test  0s {count_class(test, 0)}  1s {count_class(test, 1)}')

get_data_stats('data/decompensation', dataset='mimic-iii')
get_data_stats('dataiv/decompensation', dataset='mimic-iv')