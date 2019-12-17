# preparation/theorytab.pyで定義したTheorytabDatasetとTheorytabDataLoader

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from attrdict import AttrDict

class TheorytabDataset(Dataset):
    def __init__(self, bundles):
        self.data = bundles
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        melody = self.data[idx]['melody']
        chord = self.data[idx]['chord']    
        meta = self.data[idx]['meta']
        return melody, chord, meta
    
    def split(self, rates, shuffle=False):
        assert(sum(rates) == 1.)
        
        all_data_ids = list(range(len(self)))
        if shuffle:
            random.shuffle(all_data_ids)
        
        sizes = [int(len(self) * rate) for rate in rates[:-1]]
        sizes.append(len(self) - sum(sizes))
        
        partial_dataset_list = []
        last_right = 0
        for size in sizes:
            left, right = last_right, last_right+size
            last_right += size
            partial_data = [self.data[idx] for idx in all_data_ids[left:right]]
            dataset = TheorytabDataset(partial_data)
            partial_dataset_list.append(dataset)
        
        return partial_dataset_list

class TheorytabDataLoader:
    def __init__(self, theorytab_dataset, batch_size, shuffle=False):
        self.dataset = theorytab_dataset
        self.idx_list = list(range(len(self.dataset)))
        assert(batch_size >= 1)
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))
    
    def make_batch(self, melody_batch, chord_batch, meta_batch):
        # 一度np.arrayをかませてからLongTensorに渡すと爆速になったので採用
        # しかしバッチサイズが1000くらい大きくならないと同じ速度にならない
        melody_batch, chord_batch = np.array(melody_batch), np.array(chord_batch)
        
        if 'mnp' in meta_batch[0]:
            batch = {
                'melody': torch.FloatTensor(melody_batch),
                'chord': torch.FloatTensor(chord_batch),
                'mnp_steps': torch.LongTensor(np.array([meta['mnp']['steps'] for meta in meta_batch])),
                'mnp_weights': torch.ShortTensor(np.array([meta['mnp']['weights'] for meta in meta_batch])),
                'mnp_labels': torch.ShortTensor(np.array([meta['mnp']['labels'] for meta in meta_batch])),
                'meta': meta_batch # AttrDictを通すとlistではなくtupleになるので注意
            }
        else:
            batch = {
                'melody': torch.FloatTensor(melody_batch),
                'chord': torch.FloatTensor(chord_batch),
                'meta': meta_batch # AttrDictを通すとlistではなくtupleになるので注意
            }
        return AttrDict(batch)
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.idx_list)

        melody_batch, chord_batch, meta_batch = [], [], []
        for idx in self.idx_list:
            melody, chord, meta = self.dataset[idx]
            melody_batch.append(melody)
            chord_batch.append(chord)
            meta_batch.append(meta)

            if len(melody_batch) >= self.batch_size:
                yield self.make_batch(melody_batch, chord_batch, meta_batch)
                melody_batch, chord_batch, meta_batch = [], [], []

        if melody_batch and chord_batch and meta_batch:
            yield self.make_batch(melody_batch, chord_batch, meta_batch)
    
    def __call__(self):
        """
        何でもいいからバッチが欲しい時使う
        ジェネレータを返すわけではないので注意
        """
        return next(self.__iter__())


class TheorytabDataLoader_list:
    def __init__(self, theorytab_dataset, batch_size, shuffle=False):
        self.dataset = theorytab_dataset
        self.idx_list = list(range(len(self.dataset)))
        assert(batch_size >= 1)
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))
    
    def make_batch(self, melody_batch, chord_batch, meta_batch):
        # 一度np.arrayをかませてからLongTensorに渡すと爆速になったので採用
        # しかしバッチサイズが1000くらい大きくならないと同じ速度にならない
        melody_batch, chord_batch = np.array(melody_batch), np.array(chord_batch)
        
        if 'mnp' in meta_batch[0]:            
            steps_batch = np.array([meta['mnp']['steps'] for meta in meta_batch])
            weights_batch = np.array([meta['mnp']['weights'] for meta in meta_batch])
            labels_batch = np.array([meta['mnp']['labels'] for meta in meta_batch])
            
            batch = {
                'melody': torch.FloatTensor(melody_batch),
                'chord': torch.FloatTensor(chord_batch),
                'mnp_steps': torch.LongTensor(steps_batch),
                'mnp_weights': torch.ShortTensor(weights_batch),
                'mnp_labels': torch.ShortTensor(labels_batch),
                'meta': meta_batch # AttrDictを通すとlistではなくtupleになるので注意
            }
        else:
            batch = {
                'melody': torch.FloatTensor(melody_batch),
                'chord': torch.FloatTensor(chord_batch),
                'meta': meta_batch # AttrDictを通すとlistではなくtupleになるので注意
            }

        return AttrDict(batch)
            
    def __call__(self):
        if self.shuffle:
            random.shuffle(self.idx_list)

        batch_list = []
        melody_batch, chord_batch, meta_batch = [], [], []
        
        for idx in self.idx_list:
            melody, chord, meta = self.dataset[idx]
            melody_batch.append(melody)
            chord_batch.append(chord)
            meta_batch.append(meta)

            if len(melody_batch) >= self.batch_size:
                batch = self.make_batch(melody_batch, chord_batch, meta_batch)
                batch_list.append(batch)
                melody_batch, chord_batch, meta_batch = [], [], []

        if melody_batch and chord_batch and meta_batch:
            batch = self.make_batch(melody_batch, chord_batch, meta_batch)
            batch_list.append(batch)
        
        return batch_list