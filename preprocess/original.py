import os, random
import numpy as np
import pandas as pd
from pypianoroll import Multitrack, Track
from multiprocessing import Pool
from bundle import Bundle, PypianorollBundler
from .processor import BundlesProcessor, SequentialBundlesProcessor

class RemoveEmptyBundles(BundlesProcessor):
    def process_bundle(self, bundle):
        melody_is_empty = not bundle.melody.any()
        chord_is_empty = not bundle.chord.any()
        
        if melody_is_empty or chord_is_empty:
            return None
        
        return bundle


class RemoveShortBundles(BundlesProcessor):
    """
    初期化引数threshold_barsより小さい小節数のBundleを削除
    """
    def __init__(self, threshold_bars=4):
        self.threshold_bars = threshold_bars

    def process_bundle(self, bundle):
        if bundle.meta["original_bars"] < self.threshold_bars:
            return None
        return bundle


class Binarize(BundlesProcessor):
    """
    melody, chordをboolへ変換
    ベロシティが0ならFalse,1以上ならTrueとなる
    """
    def process_bundle(self, bundle):
        bundle.melody = bundle.melody.astype(bool)
        bundle.chord = bundle.chord.astype(bool)
        return bundle


class DownBeatResolution(BundlesProcessor):
    """
    melody, chordのbeat_resolutionを下げる
    metaのbeat_resolutionも更新される
    """
    def __init__(self, resolution_to, resolution_from=None, fill_mode=False):
        self.res_to = resolution_to
        self.fill_mode = fill_mode

    def down_resolution(self, nproll, step_width):
        result = nproll[::step_width]
        if not self.fill_mode:
            note_existence = nproll.astype(bool)
            for s in range(1, step_width):
                result *= note_existence[s::step_width]
        return result
    
    def process_bundle(self, bundle):
        res_from = bundle.meta['beat_resolution']
        
        if self.res_to >= res_from:
            raise ValueError(f"Target resolution ({self.res_to}) must be \
                             smaller than the original resolution ({res_from})")
        
        step_width = res_from // self.res_to
        
        bundle.melody = self.down_resolution(bundle.melody, step_width)
        bundle.chord = self.down_resolution(bundle.chord, step_width)
        bundle.meta['beat_resolution'] = self.res_to
        
        return bundle


class Transpose(BundlesProcessor):
    """
    to_keyはメジャースケールにおける主音
    例えば to_key=0 のとき, C minor を渡せば A minor へトランスポーズする
    """
    def __init__(self, from_key_name="C Major", to_key=0, standard_pitch_range=[36,99]):
        self.set_key_names_dict()
        self.set_key(from_key_name, to_key)
        self.std_bottom, self.std_top = standard_pitch_range
    
    def set_key_names_dict(self):
        self.sharp_key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.flat_key_names  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        self.key_names_dict = {}
        for i, (name_s, name_f) in enumerate(zip(self.sharp_key_names, self.flat_key_names)):
            self.key_names_dict[name_s], self.key_names_dict[name_f] = i, i
        
    def set_key(self, key_name, to_key=0):
        key_name, mode = key_name.split(' ')
        self.from_key = self.key_names_dict[key_name]
        self.to_key = (to_key - 3) % 12 if mode == "minor" else to_key
        key_name_list = self.sharp_key_names if '#' in key_name else self.flat_key_names
        self.to_key_name = ' '.join([key_name_list[self.to_key], mode])
        
    def get_active_range(self, nproll):
        pitches = np.where(nproll)[1]
        return np.min(pitches), np.max(pitches)
    
    def process_nproll(self, nproll):
        # standatd_pitch_rangeで移動後のlowestとhighestを囲った時，マージンの大きい方へトランスポーズする
        lowest, highest = self.get_active_range(nproll)
        shift_down      = (self.from_key - self.to_key) % 12
        shift_up        = (self.to_key - self.from_key) % 12
        margin_bottom   = (lowest - shift_down) - self.std_bottom
        margin_top      = self.std_top - (highest + shift_up)
        shift           = shift_up if margin_top > margin_bottom else -shift_down
        
        if shift != 0:
            transposed = np.zeros_like(nproll, dtype=nproll.dtype)
            transposed[:, lowest+shift:highest+shift+1] = nproll[:, lowest:highest+1]
            return transposed
        
        return nproll
            
    def process_bundle(self, bundle):
        from_key_name = bundle.meta.get('key', bundle.meta['original_key'])
        self.set_key(from_key_name, to_key=self.to_key)
        
        bundle.melody = self.process_nproll(bundle.melody)
        bundle.chord = self.process_nproll(bundle.chord)
        bundle.meta['key'] = self.to_key_name
        
        return bundle


class TrimMelodyInRange(BundlesProcessor):
    def __init__(self, octave_size, keep_size=False):
        assert(octave_size > 0)
        self.octave_size = octave_size
        self.keep_size = keep_size
    
    def process_bundle(self, bundle):
        nproll = bundle.melody
        
        oct_size = self.octave_size
        
        # get_pitch_range
        steps, pitches = np.where(nproll)
        lowest, highest = np.min(pitches), np.max(pitches)
        
        # 最も範囲内の音符が多くなるオクターブを見つける
        lowest_oct, highest_oct = lowest // 12, highest // 12
        opt_oct, opt_score = lowest_oct, 0
        for lower_oct in range(lowest_oct, highest_oct + 1 - oct_size + 1):
            oct_score = np.sum((lower_oct * 12 <= pitches) & (pitches < (lower_oct + oct_size) * 12))
            if oct_score > opt_score:
                opt_oct, opt_score = lower_oct, oct_score
        
        opt_low, opt_high = opt_oct * 12, (opt_oct + oct_size) * 12
        
        # それでも範囲外に出てしまう音符を近い方のオクターブに収める
        bottom_saturation = opt_low - lowest
        top_saturation = highest - opt_high + 1
        
        if bottom_saturation > 0:
            is_saturated = pitches < opt_low
            saturated_steps, saturated_pitches = steps[is_saturated], pitches[is_saturated]
            saturated_new_pitches = saturated_pitches % 12 + opt_low
            nproll[saturated_steps, saturated_new_pitches] = nproll[saturated_steps, saturated_pitches]
            nproll[saturated_steps, saturated_pitches] = 0
            
        if top_saturation > 0:
            is_saturated = pitches >= opt_high
            saturated_steps, saturated_pitches = steps[is_saturated], pitches[is_saturated]
            saturated_new_pitches = saturated_pitches % 12 + (opt_high - 12)
            nproll[saturated_steps, saturated_new_pitches] = nproll[saturated_steps, saturated_pitches]
            nproll[saturated_steps, saturated_pitches] = 0
        
        if self.keep_size:
            bundle.meta['melody_offset'] = 0
            bundle.meta['melody_pitch_range'] = [0, nproll.shape[1]]
        else:
            nproll = nproll[:, opt_low:opt_high]
            bundle.meta['melody_offset'] = opt_low
            bundle.meta['melody_pitch_range'] = [0, oct_size * 12]
            
        bundle.meta['melody_vocab_size'] = bundle.meta['melody_pitch_range'][1]
        
        bundle.melody = nproll
        
        return bundle


class AddSpecialPitchesToMelody(BundlesProcessor):
    def __init__(self, pad=-1, mask=-2, rest=-3):
        self.pitch_pad  = pad
        self.pitch_mask = mask
        self.pitch_rest = rest
        self.special_pitch_num = np.array([p is not None for p in [pad, mask, rest]]).sum()
    
    def set_rest_pitch(self, nproll, pitch):
        nproll[:, pitch] = nproll.any(axis=1) == False
        return nproll
    
    def process_bundle(self, bundle):
        melody = bundle.melody
        step_len, pitch_len = melody.shape
        
        extended_melody = np.zeros([step_len, pitch_len + self.special_pitch_num], dtype=melody.dtype)
        extended_melody[:, :pitch_len] = melody
        if self.pitch_rest is not None:
            extended_melody = self.set_rest_pitch(extended_melody, pitch_len)
        
        bundle.meta['melody_pitch_pad'] = self.pitch_pad
        bundle.meta['melody_pitch_mask'] = self.pitch_mask
        bundle.meta['melody_pitch_rest'] = self.pitch_rest
        bundle.meta['melody_vocab_size'] = extended_melody.shape[1]
        
        bundle.melody = extended_melody
        return bundle


class TranslateChordIntoPitchClasses(BundlesProcessor):
    def __init__(self, offset_octave=4, constant_velocity=100):
        self.offset = 12 * offset_octave
        self.constant_velocity = constant_velocity
    
    def get_velocity(self, dtype):
        if dtype == bool:
            return 1
        else:
            return self.constant_velocity
        
    def process_bundle(self, bundle):
        chord = bundle.chord
        step_len, pitch_len = chord.shape
        
        chord_pcs = np.zeros([step_len, 12], dtype=chord.dtype)
        steps, pitches = np.where(chord)
        chord_pcs[steps, pitches % 12] = self.get_velocity(chord.dtype)
        
        bundle.chord = chord_pcs
        bundle.meta['chord_offset'] = self.offset
        bundle.meta['chord_pitch_range'] = [0, 12]
        bundle.meta['chord_vocab_size'] = chord_pcs.shape[1]
        
        return bundle


class PermeateChord(BundlesProcessor):
    def process_bundle(self, bundle):
        pitch_range = bundle.meta.get('chord_pitch_range', [0,128])
        chord = bundle.chord[:, pitch_range[0]:pitch_range[1]]
        
        # 最初に音が鳴るステップを取得
        for i, step in enumerate(chord):
            if step.any():
                last_step = chord[i]
                break
        
        for step in range(len(chord)):
            # IF current ⊆ last THEN current ← last ELSE last ← current
            if np.any(chord[step] & ~last_step):
                last_step = chord[step]
            else:
                chord[step] = last_step
        
        bundle.chord = chord
        return bundle


class AddSpecialPitchesToChord(BundlesProcessor):
    def __init__(self, pad=-1):
        self.pitch_pad = pad
    
    def process_bundle(self, bundle):
        chord = bundle.chord
        step_len, pitch_len = chord.shape
        
        extended_chord = np.zeros([step_len, pitch_len + 1], dtype=chord.dtype)
        extended_chord[:, :pitch_len] = chord
        
        bundle.chord = extended_chord
        bundle.meta['chord_pitch_pad'] = self.pitch_pad
        bundle.meta['chord_vocab_size'] = extended_chord.shape[1]
        
        return bundle


class Padding(BundlesProcessor):
    def __init__(self, bar_num=16, melody_pad_pitch=-1, chord_pad_pitch=-1, constant_velocity=100):
        self.bar_num = bar_num
        self.melody_pad_pitch = melody_pad_pitch
        self.chord_pad_pitch = chord_pad_pitch
        self.constant_velocity = constant_velocity
    
    def get_velocity(self, dtype):
        if dtype == bool:
            return 1
        else:
            return self.constant_velocity
    
    def process_bundle(self, bundle):
        melody, chord = bundle.melody, bundle.chord
        melody_len, melody_pitches = melody.shape
        chord_len, chord_pitches = chord.shape
        bundle_len = max(melody_len, chord_len)
        
        beats_in_bar = bundle.meta.get('beats_in_bar', 4)
        beat_res = bundle.meta.get('beat_resolution', 24)
        bar_res = beats_in_bar * beat_res
        song_len = bar_res * self.bar_num
        
        extended_melody = np.zeros([song_len, melody_pitches], dtype=melody.dtype)
        extended_chord = np.zeros([song_len, chord_pitches], dtype=chord.dtype)
        
        extended_melody[:melody_len, :] = melody
        extended_chord[:chord_len, :] = chord

        extended_melody[bundle_len:, self.melody_pad_pitch] = self.get_velocity(melody.dtype)
        extended_chord[bundle_len:, self.chord_pad_pitch] = self.get_velocity(chord.dtype)
        
        bundle.melody = extended_melody
        bundle.chord = extended_chord
        bundle.meta['bars'] = self.bar_num
        
        return bundle


class MelodyPianorollToPitchID(BundlesProcessor):
    """
    Embedできるようにするため，ピアノロールのピッチをIDへ変換
    
    Initialize Configs
    ------
    default_id: int = 0
        ステップ内のどの音程も鳴っていない，または和音に鳴っているなどの事態の時設定されるID
    poliphonic_mode: str = 'last'
        IDに変換するうえで，1ステップに複数の音符があった場合の対処法
        'last'(default): 1つ前のピッチIDを設定．最初の場合はdefault_idを設定
        'default': default_idを設定
        'highest': ピッチの中で最も高い音符を設定
        'lowest': ピッチの中で最も低い音符を設定
    
    Output Bundle
    -------
    melodyがID列へ変換されたBundle
    """
    def __init__(self, default_id=0, poliphonic_mode='last'):
        self.default_id = default_id
        self.poliphonic_mode = poliphonic_mode
    
    def process_bundle(self, bundle):
        if bundle.meta.get('melody_is_ids', False) == True:
            return bundle
        
        step_len, pitch_num = bundle.melody.shape
        pitches_in_step = {}
        pitch_ids = np.full(step_len, self.default_id)
        for step, pitch in zip(*np.where(bundle.melody)):
            if step in pitches_in_step:
                if self.poliphonic_mode == 'default':
                    pitch_ids[step] = self.default_id
                elif self.poliphonic_mode == 'highest':
                    pitch_ids[step] = max(pitch, *pitches_in_step[step])
                elif self.poliphonic_mode == 'lowest':
                    pitch_ids[step] = min(pitch, *pitches_in_step[step])
                elif step == 0:
                    pitch_ids[step] = self.default_id
                else:
                    pitch_ids[step] = pitch_ids[step - 1]
                pitches_in_step[step].append(pitch_ids[step])
            else:
                pitch_ids[step] = pitch
                pitches_in_step[step] = [pitch]
        
        bundle.melody = pitch_ids
        bundle.meta['melody_vocab_size'] = pitch_num
        bundle.meta['melody_is_ids'] = True
        
        return bundle


class ChordVectorToChordID(BundlesProcessor):
    """
    Embedできるようにするため，コードの構成音ピッチクラスをIDへ変換
    """
    def __init__(self, pad=12):
        self.pad_id = pad
    
    def process_bundle(self, bundle):
        if bundle.meta.get('chord_is_ids', False) == True:
            return bundle
        
        chord = bundle.chord.astype(bool)
        step_len, pitch_num = chord.shape
        chord_ids = np.zeros(step_len, dtype=int)
        
        for pitch in range(pitch_num):
            if pitch != self.pad_id:
                chord_ids += chord[:, pitch] * (2 ** pitch)
            else:
                chord_ids[chord[:, self.pad_id]] = 2 ** pitch
                
        bundle.chord = chord_ids
        bundle.meta['chord_vocab_size'] = 2 ** (pitch_num - 1) + 1
        bundle.meta['chord_is_ids'] = True
        bundle.meta['chord_pitch_pad'] = 2 ** self.pad_id
        
        return bundle


class MonophonizeMelody(BundlesProcessor):
    """
    メロディが和音になっているものがあることがわかったので直す
    
    Initialize Configs
    ------
    default_id: int = 0
        ステップ内のどの音程も鳴っていない，または和音に鳴っているなどの事態の時設定されるID
    mode: str = 'last'
        IDに変換するうえで，1ステップに複数の音符があった場合の対処法
        'default': default_idを設定
        'highest'(default): ピッチの中で最も高い音符を設定
        'lowest': ピッチの中で最も低い音符を設定
    
    Output Bundle
    -------
    melodyが単音化されたBundle
    """
    def __init__(self, default_id=0, mode='highest'):
        self.default_id = default_id
        assert(mode in ['default', 'highest', 'lowest'])
        self.mode = mode
    
    def process_bundle(self, bundle):
        poly_nums = (bundle.melody > 0).sum(axis=1)
        poly_steps = np.where(poly_nums > 1)[0]
        
        if len(poly_steps) < 1:
            return bundle
        
        eye = np.eye(bundle.melody.shape[1])
        
        if self.mode == 'default':
            bundle.melody[poly_steps] = eye[self.default_id]
        else:
            poly_vecs = bundle.melody[poly_steps]
            poly_vec_steps, poly_vec_pitches = np.where(poly_vecs)
            
            target_idxs = []
            last_step = 0
            for idx, step in enumerate(poly_vec_steps):
                if last_step != step:
                    if self.mode == 'highest':
                        # highestはwhereのstepsの変化点の前の音程
                        target_idxs.append(idx - 1)
                    else:
                        # lowestはwhereのstepsの変化点の後の音程
                        target_idxs.append(idx)
                last_step = step
            
            target_idxs.append(len(poly_vec_steps)-1)
            monophonized_pitches = poly_vec_pitches[target_idxs]
            bundle.melody[poly_steps] = eye[monophonized_pitches]
        
        return bundle


class AddNoteAreasToMeta(BundlesProcessor):
    def process_bundle(self, bundle):
        note_areas = []
        last_pitch, last_step = None, None
        lowest, highest = bundle.meta['melody_pitch_range']
        rest_id = bundle.meta['melody_pitch_rest']
        
        if bundle.meta.get('melody_is_ids', False) == True:
            melody = bundle.melody
        else:
            steps, pitches = np.where(bundle.melody)

        for step, pitch in zip(steps, pitches):
            if last_pitch != pitch or last_step+1 != step:
                note_areas.append([])
            note_areas[-1].append(step)
            last_pitch = pitch
            last_step = step

        bundle.meta['note_areas'] = note_areas

        return bundle


INPUT_DIR = None
stop_by_exception = True
def preprocess_theorytab(row, sequential_processor):
    if type(row) == tuple:
        row = row[1]
    elif type(row) == pd.core.frame.DataFrame:
        row = row.iloc[0]
    path = row.path
        
    try:
        ppr = Multitrack(os.path.join(INPUT_DIR, path))
        bundles = PypianorollBundler().bundle(ppr, row)
        bundles = sequential_processor(bundles)
    except Exception as e:
        if stop_by_exception: raise e
        return f"ERROR: {path}" # Means Error
    
    bundles = [bundle.get_dict() for bundle in bundles]
    
    return bundles # Valid Result even empty


def preprocess_theorytab_original(row):
    sequential_processor = SequentialBundlesProcessor(processors=[
        RemoveEmptyBundles(),
        RemoveShortBundles(),
        Binarize(),
        DownBeatResolution(resolution_to=12),
        Transpose(),
        MonophonizeMelody(),
        TrimMelodyInRange(octave_size=2),
        AddSpecialPitchesToMelody(rest=24, mask=25, pad=26),
        TranslateChordIntoPitchClasses(),
        PermeateChord(),
        AddSpecialPitchesToChord(pad=12),
        Padding(),
        AddNoteAreasToMeta(),
    ])
    return preprocess_theorytab(row, sequential_processor)


def load_bundle_list(input_csv, input_dir, core_num=1):
    while os.getcwd().split('/')[-1] != 'schwbert': os.chdir('..')

    global INPUT_DIR
    INPUT_DIR = input_dir
    
    df = pd.read_csv(input_csv)
    df = df[(df["nokey"] == False) & (df["time_signature"] == "4/4")]
    df = df[df["has_melody_track"] & df["has_chord_track"]]
    df = df.reset_index()
    process_func = preprocess_theorytab_original
    
    print("start preprocessing...")
    pool = Pool(core_num)
    if core_num > 1:
        result_bundles_list = pool.map(process_func, df.iterrows())
    else:
        result_bundles_list = list(map(process_func, df.iterrows()))
    
    pool.close()
    
    print("unpacking result bundles list...")
    bundle_list = []
    error_list = []
    for bundles in result_bundles_list:
        if type(bundles) is str:
            error_list.append(bundles.split(' ')[-1])
        else:
            for bundle in bundles:
                bundle_list.append(bundle)

    print("Preprocessing Finished!")
    print(f"load {len(bundle_list)} bundles")
    print(f"{len(error_list)} error(s) occured")
    
    if error_list:
        print("error files list")
        for e in error_list:
            print(e)
    
    return bundle_list