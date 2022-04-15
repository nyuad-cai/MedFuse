#!/usr/bin/python

# import thread
import time
from PIL import Image
import glob
from tqdm import tqdm
import os

print('starting')
data_dir = 'physionet.org/files/mimic-cxr-jpg'
version = '2.0.0'

paths_done = glob.glob(f'{data_dir}/{version}/resized/**/*.jpg', recursive = True)
print('done', len(paths_done))

paths_all = glob.glob(f'{data_dir}/{version}/files/**/*.jpg', recursive = True)
print('all', len(paths_all))



done_files = [os.path.basename(path) for path in paths_done]

paths = [path for path in paths_all if os.path.basename(path) not in done_files ]
print('left', len(paths))

def resize_images(path):
    basewidth = 512
    filename = path.split('/')[-1]
    img = Image.open(path)

    wpercent = (basewidth/float(img.size[0]))
    
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize))
    
    img.save(f'{data_dir}/{version}/resized/{filename}')


from multiprocessing.dummy import Pool as ThreadPool

threads = 10

for i in tqdm(range(0, len(paths), threads)):
    paths_subset = paths[i: i+threads]
    pool = ThreadPool(len(paths_subset))
    pool.map(resize_images, paths_subset)
    pool.close()
    pool.join()
