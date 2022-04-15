from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from mimic3models import common_utils
import threading
import random
import os


import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
# import 
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



# read_by_file_name

class EHRdataset(Dataset):
    def __init__(self, discretizer, normalizer, listfile, dataset_dir, return_names=True):
        # self.batch_size = batch_size
        # self.target_repl = target_repl
        # self.shuffle = shuffle
        self.return_names = return_names
        # self.reader = reader
        self.discretizer = discretizer
        self.normalizer = normalizer


        self._dataset_dir = dataset_dir
        # self._current_index = 0
        # if listfile is None:
        #     listfile_path = os.path.join(dataset_dir, "listfile.csv")
        # else:
        listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        # import pdb; pdb.set_trace()
        self.classes = self._listfile_header.strip().split(',')[3:]
        self._data = self._data[1:]



        self._data = [line.split(',') for line in self._data]
        # import pdb; pdb.set_trace()
        self.data_map = {
            mas[0]: {
                'labels': list(map(int, mas[3:])),
                'stay_id': float(mas[2]),
                'time': float(mas[1]),
                }
                for mas in self._data
        }

        self.names = list(self.data_map.keys())
    
    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)
    def read_by_file_name(self, index):

        # if index < 0 or index >= len(self._data):
        #     raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")
        
        # timeseries_path = self.ehr_paths[index]

        
        
        t = self.data_map[index]['time']
        y = self.data_map[index]['labels']
        stay_id = self.data_map[index]['stay_id']
        # name = self._data[index][0]
        # t = self._data[index][1]
        # y = self._data[index][2]
        (X, header) = self._read_timeseries(index)

        return {"X": X,
                "t": t,
                "y": y,
                'stay_id': stay_id,
                "header": header,
                "name": index}

        
    def __getitem__(self, index):
        # pass
        # N = reader.get_number_of_examples()
        # if small_part:
        #     N = 200

        # self.reader.read_next()
        if isinstance(index, int):
            index = self.names[index]
        ret = self.read_by_file_name(index)
        data = ret["X"]
        ts = ret["t"]
        ys = ret["y"]
        names = ret["name"]
        # import pdb; pdb.set_trace()
        data = self.discretizer.transform(data, end=ts)[0] 
        # for (X, t) in zip(data, ts)]
        if (self.normalizer is not None):
            data = self.normalizer.transform(data)
            #  for X in data]
        ys = np.array(ys, dtype=np.int32)
        # self.data = (data, ys)
        # self.ts = ts
        # self.names = names
        #  = self.filenames_loaded[index]
        # import pdb; pdb.set_trace()
        
        # img = Image.open(self.cxr_paths[self.cxr_file_names[index]]).convert('RGB')
        # ehr = pd.read_csv(self.ehr_paths[self.ehr_file_names[index]])

        # labels = torch.tensor(self.labels[index]).float()

        # if self.transform is not None:
        #     img = self.transform(img)
        # return img, labels

        return data, ys

    
    def __len__(self):
        return len(self.names)



def get_data_loader(discretizer, normalizer, dataset_dir, batch_size):
    train_ds = EHRdataset(discretizer, normalizer, 'data/mimic-iv-data/phenotyping/train_listfile.csv', dataset_dir)
    val_ds = EHRdataset(discretizer, normalizer, 'data/mimic-iv-data/phenotyping/val_listfile.csv', dataset_dir)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16)

    return train_dl, val_dl
    # train_ds = EHRdataset(discretizer, normalizer, listfile, dataset_dir)
        
def my_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    x = [item[0] for item in batch]

    x, seq_length = common_utils.pad_zeros(x)

    # data = pack_sequence(data, enforce_sorted=False)
    targets = np.array([item[1] for item in batch])
    return [x, targets, seq_length]

class BatchGen(object):

    def __init__(self, reader, discretizer, normalizer, batch_size,
                 small_part, target_repl, shuffle, return_names=True):
        self.batch_size = batch_size
        self.target_repl = target_repl
        self.shuffle = shuffle
        self.return_names = return_names

        self._load_data(reader, discretizer, normalizer, small_part)

        self.steps = (len(self.data[0]) + batch_size - 1) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _load_data(self, reader, discretizer, normalizer, small_part=False):
        N = reader.get_number_of_examples()
        if small_part:
            N = 200
        ret = common_utils.read_chunk(reader, N)
        data = ret["X"]
        ts = ret["t"]
        ys = ret["y"]
        names = ret["name"]
        data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if (normalizer is not None):
            data = [normalizer.transform(X) for X in data]
        ys = np.array(ys, dtype=np.int32)
        self.data = (data, ys)
        self.ts = ts
        self.names = names

    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                N = len(self.data[1])
                order = list(range(N))
                random.shuffle(order)
                tmp_data = [[None] * N, [None] * N]
                tmp_names = [None] * N
                tmp_ts = [None] * N
                for i in range(N):
                    tmp_data[0][i] = self.data[0][order[i]]
                    tmp_data[1][i] = self.data[1][order[i]]
                    tmp_names[i] = self.names[order[i]]
                    tmp_ts[i] = self.ts[order[i]]
                self.data = tmp_data
                self.names = tmp_names
                self.ts = tmp_ts
            else:
                # sort entirely
                X = self.data[0]
                y = self.data[1]
                (X, y, self.names, self.ts) = common_utils.sort_and_shuffle([X, y, self.names, self.ts], B)
                self.data = [X, y]

            self.data[1] = np.array(self.data[1])  # this is important for Keras
            for i in range(0, len(self.data[0]), B):
                x = self.data[0][i:i+B]
                y = self.data[1][i:i+B]
                names = self.names[i:i + B]
                ts = self.ts[i:i + B]

                x, seq_length = common_utils.pad_zeros(x)
                y = np.array(y)  # (B, 25)

                if self.target_repl:
                    y_rep = np.expand_dims(y, axis=1).repeat(x.shape[1], axis=1)  # (B, T, 25)
                    batch_data = (x, [y, y_rep], seq_length)
                else:
                    batch_data = (x, y, seq_length)

                if not self.return_names:
                    yield batch_data
                else:
                    yield {"data": batch_data, "names": names, "ts": ts}

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()


def save_results(names, ts, predictions, labels, path):
    n_tasks = 25
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        header = ["stay", "period_length"]
        header += ["pred_{}".format(x) for x in range(1, n_tasks + 1)]
        header += ["label_{}".format(x) for x in range(1, n_tasks + 1)]
        header = ",".join(header)
        f.write(header + '\n')
        for name, t, pred, y in zip(names, ts, predictions, labels):
            line = [name]
            line += ["{:.6f}".format(t)]
            line += ["{:.6f}".format(a) for a in pred]
            line += [str(a) for a in y]
            line = ",".join(line)
            f.write(line + '\n')
