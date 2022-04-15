import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class EHRdataset(Dataset):
    def __init__(self, discretizer, normalizer, listfile, dataset_dir, return_names=True, period_length=48.0):
        self.return_names = return_names
        self.discretizer = discretizer
        self.normalizer = normalizer
        self._period_length = period_length

        self._dataset_dir = dataset_dir
        listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self.CLASSES = self._listfile_header.strip().split(',')[3:]
        self._data = self._data[1:]



        self._data = [line.split(',') for line in self._data]
        self.data_map = {
            mas[0]: {
                'labels': list(map(float, mas[3:])),
                'stay_id': float(mas[2]),
                'time': float(mas[1]),
                }
                for mas in self._data
        }

        # import pdb; pdb.set_trace()

        # self._data = [(line_[0], float(line_[1]), line_[2], float(line_[3])  ) for line_ in self._data]



        self.names = list(self.data_map.keys())
    
    def _read_timeseries(self, ts_filename, time_bound=None):
        
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                if time_bound is not None:
                    t = float(mas[0])
                    if t > time_bound + 1e-6:
                        break
                ret.append(np.array(mas))
        return (np.stack(ret), header)
    
    def read_by_file_name(self, index, time_bound=None):
        t = self.data_map[index]['time'] if time_bound is None else time_bound
        y = self.data_map[index]['labels']
        stay_id = self.data_map[index]['stay_id']
        (X, header) = self._read_timeseries(index, time_bound=time_bound)

        return {"X": X,
                "t": t,
                "y": y,
                'stay_id': stay_id,
                "header": header,
                "name": index}

    def get_decomp_los(self, index, time_bound=None):
        # name = self._data[index][0]
        # time_bound = self._data[index][1]
        # ys = self._data[index][3]

        # (data, header) = self._read_timeseries(index, time_bound=time_bound)
        # data = self.discretizer.transform(data, end=time_bound)[0] 
        # if (self.normalizer is not None):
        #     data = self.normalizer.transform(data)
        # ys = np.array(ys, dtype=np.int32) if len(ys) > 1 else np.array(ys, dtype=np.int32)[0]
        # return data, ys

        # data, ys = 
        return self.__getitem__(index, time_bound)


    def __getitem__(self, index, time_bound=None):
        if isinstance(index, int):
            index = self.names[index]
        ret = self.read_by_file_name(index, time_bound)
        data = ret["X"]
        ts = ret["t"] if ret['t'] > 0.0 else self._period_length
        ys = ret["y"]
        names = ret["name"]
        data = self.discretizer.transform(data, end=ts)[0] 
        if (self.normalizer is not None):
            data = self.normalizer.transform(data)
        ys = np.array(ys, dtype=np.int32) if len(ys) > 1 else np.array(ys, dtype=np.int32)[0]
        return data, ys

    
    def __len__(self):
        return len(self.names)


def get_datasets(discretizer, normalizer, args):
    train_ds = EHRdataset(discretizer, normalizer, f'{args.ehr_data_dir}/{args.task}/train_listfile.csv', os.path.join(args.ehr_data_dir, f'{args.task}/train'))
    val_ds = EHRdataset(discretizer, normalizer, f'{args.ehr_data_dir}/{args.task}/val_listfile.csv', os.path.join(args.ehr_data_dir, f'{args.task}/train'))
    test_ds = EHRdataset(discretizer, normalizer, f'{args.ehr_data_dir}/{args.task}/test_listfile.csv', os.path.join(args.ehr_data_dir, f'{args.task}/test'))
    return train_ds, val_ds, test_ds

def get_data_loader(discretizer, normalizer, dataset_dir, batch_size):
    train_ds, val_ds, test_ds = get_datasets(discretizer, normalizer, dataset_dir)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16)

    return train_dl, val_dl
        
def my_collate(batch):
    x = [item[0] for item in batch]
    x, seq_length = pad_zeros(x)
    targets = np.array([item[1] for item in batch])
    return [x, targets, seq_length]

def pad_zeros(arr, min_length=None):

    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length