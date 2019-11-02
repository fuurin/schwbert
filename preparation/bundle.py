import numpy as np
import pandas as pd
from typing import List, Union
from pypianoroll import Track, Multitrack

class Bundle:
    def __init__(self, melody=None, chord=None, meta={}):
        self.melody = melody
        self.chord = chord
        self.meta = meta
    
    def get_dict(self):
        dic = {}
        dic['melody'] = self.melody
        dic['chord'] = self.chord
        dic['meta'] = self.meta
        return dic
    
    def set_dict(self, dictionary):
        self.melody = dictionary['melody']
        self.chord = dictionary['chord']
        self.meta = dictionary['meta']
        return self
    
    def get_melody_track(self, program=0):
        if self.melody is None:
            return None
        
        nproll = np.zeros([len(self.melody), 128], dtype=self.melody.dtype)
        
        bottom, top = self.meta.get('melody_pitch_range', [0, 128])
        offset = self.meta.get('melody_offset', 0)
        
        source = self.melody[:, bottom:top]
        nproll[:, offset:offset+source.shape[1]] = source
        
        return Track(nproll, program=program, name='melody')

    def get_chord_track(self, program=0):
        if self.chord is None:
            return None
        
        nproll = np.zeros([len(self.chord), 128], dtype=self.chord.dtype)
        
        bottom, top = self.meta.get('chord_pitch_range', [0, 128])
        offset = self.meta.get('chord_offset', 0)
        
        source = self.chord[:, bottom:top]
        nproll[:, offset:offset+source.shape[1]] = source
        
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

            bundles.append(Bundle(bundle_melody, bundle_chord, meta))

        return bundles
