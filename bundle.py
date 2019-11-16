import numpy as np
import pandas as pd
from typing import List, Union
from pypianoroll import Track, Multitrack

class Bundle:
    def __init__(self, bundle_dict=None, melody=np.array([]), chord=np.array([]), meta={}):
        if bundle_dict is None:
            self.melody = melody
            self.chord = chord
            self.meta = meta
        else:
            self.set_dict(bundle_dict)
    
    def __repr__(self):
        meta = "\n".join([f"{item[0]}: {item[1]}" for item in self.meta.items()])
        return f"----- bundle info -----\nmelody shape: {self.melody.shape}\nchord shape: {self.chord.shape}\n{meta}"
    
    def get_dict(self):
        dic = {}
        dic['melody'] = self.melody
        dic['chord'] = self.chord
        dic['meta'] = self.meta
        return dic
    
    def set_dict(self, dictionary):
        self.melody = dictionary.get('melody', np.array([]))
        self.chord = dictionary.get('chord', np.array([]))
        self.meta = dictionary.get('meta', {})
        return self
    
    def get_melody_track(self, program=0):
        if self.melody is None:
            return None
        
        is_ids = self.meta.get('melody_is_ids', False)
        bottom, top = self.meta.get('melody_pitch_range', [0, 128])
        offset = self.meta.get('melody_offset', 0)
        melody_vocab_size = self.meta.get('melody_vocab_size', 128)
        step_len = len(self.melody)
        
        if is_ids:
            nproll = np.zeros([step_len, 128], dtype=bool)
            nproll[np.arange(step_len), offset+self.melody] = True
            nproll[:, :offset+bottom], nproll[:, offset+top:offset+melody_vocab_size] = False, False
        else:
            nproll = np.zeros([step_len, 128], dtype=self.melody.dtype)
            nproll[:, offset:offset+top-bottom] = self.melody[:, bottom:top]
        
        return Track(nproll, program=program, name='melody')

    def get_chord_track(self, program=0):
        chord = self.chord
        if chord is None:
            return None
        
        is_ids = self.meta.get('chord_is_ids', False)
        bottom, top = self.meta.get('chord_pitch_range', [0, 128])
        offset = self.meta.get('chord_offset', 0)
        step_len = len(chord)
        
        if is_ids:
            nproll = np.zeros([step_len, 128], dtype=bool)
            for pitch in range(top - bottom):
                nproll[:, offset+pitch] = chord % 2
                chord = chord // 2
        else:
            nproll = np.zeros([step_len, 128], dtype=chord.dtype)
            nproll[:, offset:offset+top-bottom] = chord[:, bottom:top]
        
        return Track(nproll, program=program, name='chord')
    
    def get_ppr(self, melody_program=0, chord_program=0, beats_in_bar=4):
        melody = self.get_melody_track(melody_program)
        chord = self.get_chord_track(chord_program)
        
        tracks = [trk for trk in [melody, chord] if trk is not None]
        tempo = np.array([self.meta.get('bpm', 120)])
        beat_res = self.meta.get('beat_resolution', 24)
        name = self.meta.get('path', 'no_name')
        
        downbeat = np.zeros(len(tracks[0].pianoroll), dtype=bool)
        downbeat[::beat_res] = True
        
        ppr = Multitrack(tracks=tracks, tempo=tempo, beat_resolution=beat_res, name=name, downbeat=downbeat)
        
        return ppr


class PypianorollBundler:
    def __init__(self, bundle_bar_num:int=16, beats_in_bar=4):
        self.bundle_bar_num = bundle_bar_num
        self.beats_in_bar = beats_in_bar
        
    def remove_tail_space(self, ppr, bar_len):
        melody = ppr.tracks[0].pianoroll
        chord = ppr.tracks[1].pianoroll
        
        song_len = max(len(melody), len(chord))
        tail_len = song_len % bar_len
        if tail_len > 0:
            melody_tail_is_empty = not melody[-tail_len:].any()
            chord_tail_is_empty = not chord[-tail_len:].any()
            if melody_tail_is_empty and chord_tail_is_empty:
                ppr.tracks[0].pianoroll = melody[:-tail_len]
                ppr.tracks[1].pianoroll = chord[:-tail_len]
            else:
                new_song_len = (song_len - tail_len) + bar_len
                melody_len, melody_pitches = melody.shape
                chord_len, chord_pitches = chord.shape
                new_melody = np.zeros([new_song_len, melody_pitches])
                new_chord = np.zeros([new_song_len, chord_pitches])
                new_melody[:melody_len, :] = melody
                new_chord[:chord_len, :] = chord
                ppr.tracks[0].pianoroll = new_melody
                ppr.tracks[1].pianoroll = new_chord
                
        return ppr
        
    def bundle(self, ppr: Multitrack, row: Union[pd.core.series.Series, None]=None) -> List[Bundle]:
        beat_res = ppr.beat_resolution
        bar_len = beat_res * self.beats_in_bar
        
        ppr = self.remove_tail_space(ppr, bar_len)
        melody = ppr.tracks[0].pianoroll
        chord = ppr.tracks[1].pianoroll
        
        bundle_len = self.bundle_bar_num * bar_len
        bar_num = len(melody) // bar_len
        bundle_num = int(np.ceil(bar_num / self.bundle_bar_num))

        bundles = []
        for b in range(bundle_num):
            left, right = b*bundle_len, (b+1) * bundle_len
            bundle_melody = melody[left:right]
            bundle_chord = chord[left:right]
            meta = {
                'range_from': b*self.bundle_bar_num,
                'range_for': b*self.bundle_bar_num + len(bundle_melody) // bar_len,
                'original_bars': len(bundle_melody) // bar_len,
                'beat_resolution': beat_res,
                'beats_in_bar': self.beats_in_bar
            }
            
            if row is not None:
                meta['path'] = row.path
                meta['bpm'] = row.bpm
                meta['original_key'] = row.estimated_key_signature
            
            bundle_dict = {
                'melody': bundle_melody,
                'chord': bundle_chord,
                'meta': meta,
            }
            
            bundles.append(Bundle(bundle_dict))

        return bundles
