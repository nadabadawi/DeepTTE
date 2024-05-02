import time
import utils

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import ujson as json

class MySet(Dataset):
    def __init__(self, input_file):
        self.content = []
        
        with open('./new_data/' + input_file, 'r') as file:
            for line in file:
                try:
                    json_dict = json.loads(line)
                    self.content.append(json_dict)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line: {line}. Error: {e}")

        self.lengths = [len(x['lngs']) for x in self.content]

        # self.content = open('./data/' + input_file, 'r').readlines()
        # self.content = map(lambda x: json.loads(x), self.content)
        # self.lengths = map(lambda x: len(x['lngs']), self.content)

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)

def collate_fn(data):
    # print("\n\nFunction: ", len(data))
    # print("------------------")
    stat_attrs = ['dist', 'time']
    info_attrs = ['driverID', 'dateID', 'weekID', 'timeID']
    traj_attrs = ['lngs', 'lats', 'time_gap', 'dist_gap']
    # traj_attrs = ['lngs', 'lats', 'states', 'time_gap', 'dist_gap']

    attr, traj = {}, {}

    lens = np.asarray([len(item['lngs']) for item in data])

    # print("Lens: ", len(lens))

    for key in stat_attrs:
        x = torch.FloatTensor([item[key] for item in data])
        attr[key] = utils.normalize(x, key)

    # print("X: ", x)
    # print("ATRR: ", attr)

    for key in info_attrs:
        attr[key] = torch.LongTensor([item[key] for item in data])

    for key in traj_attrs:
        # pad to the max length
        # print("Over here!")
        # print([len(item[key]) for item in data])
        # seqs = np.asarray(item[key] for item in data)
        # seqs = []
        seqs = [item[key] for item in data]
        # print("FLAG: ", seqs)
        # print("Range: ", np.arange(lens.max()))
        # print("Lens: ", lens)
        # print("Check: ", lens[:, None])
        mask = np.arange(lens.max()) < lens[:, None]
        # print("mask: ", mask)
        padded = np.zeros(mask.shape, dtype = np.float32)
        # seqs = list(seqs)
        padded[mask] = np.concatenate(seqs)
        # print("Array:" , padded)

        if key in ['lngs', 'lats', 'time_gap', 'dist_gap']:
            padded = utils.normalize(padded, key)

        padded = torch.from_numpy(padded).float()
        traj[key] = padded

    lens = lens.tolist()
    traj['lens'] = lens

    return attr, traj

class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        # print(batch_size)
        self.batch_size = batch_size
        self.lengths = dataset.lengths
        self.indices = list(range(self.count))
        # print("------------------------------")
        # print(list(self.indices))

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size

def get_loader(input_file, batch_size):
    dataset = MySet(input_file = input_file)

    batch_sampler = BatchSampler(dataset, batch_size)

    data_loader = DataLoader(dataset = dataset, \
                             batch_size = 1, \
                             collate_fn = lambda x: collate_fn(x), \
                             num_workers = 4,
                             batch_sampler = batch_sampler,
                             pin_memory = True
    )

    return data_loader
