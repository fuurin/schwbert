import pandas as pd
from typing import List, Union
from pypianoroll import Multitrack

class Bundle:
    def __init__(self, melody, chord, meta):
        self.melody = melody
        self.chord = chord
        self.meta = meta
    
    def get_dict(self):
        dic = {}
        dic['melody'] = self.melody
        dic['chord'] = self.chord
        dic['meta'] = self.meta
        return dic

class PypianorollBundler:
    def __init__(self, bundle_bar_num:int=16, beats_in_bar=4):
        self.bundle_bar_num = bundle_bar_num
        self.beats_in_bar = beats_in_bar
        
    def bundle(self, ppr: Multitrack, row: Union[pd.core.series.Series, None]=None) -> List[Bundle]:
        melody_nproll = ppr.tracks[0].pianoroll
        chord_nproll = ppr.tracks[1].pianoroll
        beat_res = ppr.beat_resolution
        bar_len = beat_res * self.beats_in_bar
        bundle_len = self.bundle_bar_num * bar_len
        bar_num = ppr.get_active_length() // bar_len
        bundle_num = (bar_num // self.bundle_bar_num) + 1

        bundles = []
        for b in range(bundle_num):
            left, right = b*bundle_len, (b+1) * bundle_len
            melody = melody_nproll[left:right]
            chord = chord_nproll[left:right]
            meta = {
                'range_from': b*self.bundle_bar_num,
                'range_for': b*self.bundle_bar_num + len(melody) // bar_len,
                'original_bars': len(melody) // bar_len,
                'beat_resolution': beat_res,
                'beats_in_bar': self.beats_in_bar
            }
            
            if row is not None:
                meta['path'] = row.path
                meta['bpm'] = row.bpm
                meta['original_key'] = row.estimated_key_signature

            bundles.append(Bundle(melody, chord, meta))

        return bundles