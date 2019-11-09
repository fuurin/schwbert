# preparation/theorytab.pyで定義したTheorytabDatasetとTheorytabDataLoader

import random
import numpy as np
from torch.utils.data import Dataset
from attrdict import AttrDict

class TheorytabDataset(Dataset):
    def __init__(self, bundles, include_meta=True):
        self.data = bundles
        self.data_num = len(bundles)
        self.include_meta = include_meta
    
    def add_note_areas_to_meta(self, include_rest=True):
        # ステップを音符ごとに分ける
        for b, bundle in enumerate(self.data):
            note_areas = []
            last_pitch = None
            lowest, highest = bundle['meta']['melody_pitch_range']
            rest_id = bundle['meta']['melody_pitch_rest']
            
            for step, pitch in enumerate(bundle['melody']):
                if (lowest <= pitch <= highest) or (include_rest and pitch == rest_id):
                    if last_pitch != pitch:
                        note_areas.append([])
                    note_areas[-1].append(step)
                last_pitch = pitch
                    
            self.data[b]['meta']['note_areas'] = note_areas
    
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        melody = self.data[idx]['melody']
        chord = self.data[idx]['chord']
        
        if self.include_meta:
            meta = self.data[idx]['meta']
            return melody, chord, meta
        else:
            return melody, chord
    
    def split(self, rates, shuffle=False):
        assert(sum(rates) == 1.)
        
        all_data_ids = list(range(self.data_num))
        if shuffle:
            random.shuffle(all_data_ids)
        
        sizes = [int(self.data_num * rate) for rate in rates[:-1]]
        sizes.append(self.data_num - sum(sizes))
        
        partial_dataset_list = []
        last_right = 0
        for size in sizes:
            left, right = last_right, last_right+size
            last_right += size
            partial_data = [self.data[idx] for idx in all_data_ids[left:right]]
            dataset = TheorytabDataset(partial_data, self.include_meta)
            partial_dataset_list.append(dataset)
        
        return partial_dataset_list


class TheorytabDataLoader:
    def __init__(self, theorytab_dataset, batch_size, shuffle=False):
        self.dataset = theorytab_dataset
        self.idx_list = list(range(len(self.dataset)))
        self.include_meta = self.dataset.include_meta
        assert(batch_size >= 1)
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))
    
    def make_batch(self, melody_batch, chord_batch, meta_batch):
        # 一度np.arrayをかませてからLongTensorに渡すと爆速になったので採用
        # しかしバッチサイズが1000くらい大きくならないと同じ速度にならない
        melody_batch, chord_batch = np.array([melody_batch, chord_batch])
        batch = {
            'melody': torch.LongTensor(melody_batch),
            'chord': torch.LongTensor(chord_batch),
            'meta': meta_batch # AttrDictを通すとlistではなくtupleになるので注意
        }
        return AttrDict(batch)
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.idx_list)

        melody_batch, chord_batch, meta_batch = [], [], []
        for idx in self.idx_list:

            if self.include_meta:
                melody, chord, meta = self.dataset[idx]
                meta_batch.append(meta)
            else:
                melody, chord = self.dataset[idx]

            melody_batch.append(melody)
            chord_batch.append(chord)

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