from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re

from mimic3models.phenotyping import utils
from mimic3benchmark.readers import PhenotypingReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger
from mimic3models.pytorch_models.trainer import Trainer

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of phenotyping task',
                    default='data/mimic-iv-data/phenotyping/')
parser.add_argument('--save_dir', type=str, help='Directory relative which all output files are stored',
                    default='checkpoints')



# parser.add_argument('--save_dir', type=str, help='Directory relative which all output files are stored',
#                     default='.')
                    
args = parser.parse_args()
print(args)

try:  
    os.mkdir(args.save_dir)  
except OSError as error:
    print(error) 


if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
# _merged_listfile listfile
train_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                 listfile=os.path.join(args.data, 'train_merged_listfile.csv'))

# val_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                            #    listfile=os.path.join(args.data, 'val_merged_listfile.csv'))


# import pdb; pdb.set_trace()


# print(len(train_reader._data))
# print(len(val_reader._data))
discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')


import pdb; pdb.set_trace()
discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
# import pdb; pdb.set_trace()

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)
num_classes = 25
# args_dict = dict(args._get_kwargs())
# args_dict['header'] = discretizer_header
# args_dict['task'] = 'ph'
# args_dict['num_classes'] = num_classes
# args_dict['target_repl'] = target_repl

train_dl, val_dl = utils.get_data_loader(discretizer, normalizer, os.path.join(args.data, 'train'), args.batch_size)

# x,y,sq = next(iter(train_dl))
# import pdb; pdb.set_trace()

n_trained_chunks = 0
# train_data_gen = utils.BatchGen(train_reader, discretizer,
#                                 normalizer, args.batch_size,
#                                 args.small_part, target_repl, shuffle=True)
# val_data_gen = utils.BatchGen(val_reader, discretizer,
#                               normalizer, 10,
#                               args.small_part, target_repl, shuffle=False)

if args.mode == 'train':
    trainer = Trainer(train_dl, 
        val_dl, 
        args,
        num_classes=num_classes, 
    )
    print("==> training")
    trainer.train()
else:
    raise ValueError("not Implementation for args.mode")
