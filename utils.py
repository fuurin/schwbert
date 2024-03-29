import os, glob, time, platform, numpy as np
import torch
from torchviz import make_dot
import pretty_midi
from pretty_midi import PrettyMIDI
from pypianoroll import Multitrack, Track
from IPython.display import Audio
from scipy.io import wavfile as spw
from pydub import AudioSegment as AS
import matplotlib.pyplot as plt
from pylab import rcParams

class Timer():
    """
    with Timer():
        # 計測したい処理
        # 約 1/100000 [sec] だけこいつを使った方が遅くなることに注意
    
    with Timer(fmt="endtime: {:f}"):
        # 計測したい処理
        # このようにformatを指定することもできる
    
    """
    def __init__(self, name="Timer"):
        self.fmt = name + ': {:f}'
    
    def get_time():
        return time.time() - self.start
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, _1, _2, _3):
        end = time.time() - self.start
        print(self.fmt.format(end))



def count_params(*modules, requires_grad=True):
    param_nums = []
    for module in modules:
        for param in module.parameters():
            if param.requires_grad and requires_grad:
                param_nums.append(param.numel())
    return sum(param_nums)


def show_model(model, output):
    return make_dot(output, params=dict(model.named_parameters()))


def grad_status(module, print_out=True):
    names, avgs, stds = [], [], []
    for (parameter, name) in zip(module.parameters(), module.state_dict().keys()):
        names.append(name)
        avg = parameter.grad.mean()
        std = parameter.grad.std()
        avgs.append(avg.item() if not torch.isnan(avg) else 0)
        stds.append(std.item() if not torch.isnan(std) else 0)
    
    if print_out:
        for (name, avg, std) in zip(names, avgs, stds):
            print(f"avg: {avg:>.6f} | std: {std:>.6f} | name: {name:<70} ")
        print(f"total average: {sum(avgs) / len(avgs):<.6f}")
        print(f"std average: {sum(stds) / len(stds):<.6f}")
    else:
        return names, avgs, stds
        
        
        
class Sampler:
    def __init__(self, base_dir, dataframe):
        self.base_dir = base_dir
        self.df = dataframe
    
    def ppr_by_path(self, path, with_row=True, verbose=True):
        absolute_path = os.path.join(self.base_dir, path)
        if verbose:
            print(f"path: {path}")
        if with_row:
            row = self.df[self.df['path'] == path]
            row = row.T.squeeze()
            return Multitrack(absolute_path), row
        return Multitrack(absolute_path)
    
    def ppr(self, index=0, with_row=True, verbose=True):
        path = self.df.loc[index]['path']
        if verbose:
            print(f"id: {index}")
        return self.ppr_by_path(path, with_row=with_row, verbose=verbose)


def grid_plot(ppr, 
        bar_range=None, pitch_range='auto', 
        beats_in_bar=4, beat_resolution=24, 
        show_white_key_ticks=False, figsize=[21, 10]
    ):
    """
    pretty ploting for pypianoroll
    """
    orgSize = rcParams['figure.figsize']
    rcParams['figure.figsize'] = figsize
    
    if isinstance(ppr, Track):
        downbeat = list(range(ppr.pianoroll.shape[0]))
        ppr = Multitrack(tracks=[ppr], downbeat=downbeat, beat_resolution=beat_resolution)
    
    beat_res = ppr.beat_resolution
    bar_res = beats_in_bar * beat_res
    downbeat = ppr.downbeat
    ppr.downbeat = np.zeros_like(ppr.downbeat, dtype=bool)
    
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    major_scale_name = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    major_color = ['red', 'orange', 'yellow', 'green', 'cyan', 'mediumblue', 'magenta']
    major = list(zip(major_scale, major_color))
    
    fig, axs = ppr.plot(xtick="beat")
    for a, ax in enumerate(axs):
        ax.grid(False)
        ax.xaxis.set_ticks_position('none')
        ax.set_xticklabels([], minor=True)
        ax.set_xticklabels(range(len(ppr.downbeat) // beat_res), minor=False)
        
        # pretty_midiに合わせてC-1を0とする
        if show_white_key_ticks:
            ax.set_yticks([k+12*i for i in range(11) for k in major_scale][:75])
            ax.set_yticklabels([k+str(i-1) for i in range(11) for k in major_scale_name][:75])
        else:
            ax.set_yticklabels([f'C{i - 1}' for i in range(11)])
        
        xlim = ax.get_xlim()
        if bar_range:
            xlim = (bar_range[0] * bar_res, bar_range[1] * bar_res - 0.5)
        ax.set_xlim(*xlim)
        
        if pitch_range == 'auto':
            try:
                low, high = ppr.tracks[a].get_active_pitch_range()
            except ValueError:
                low, high = 66, 66
            ax.set_ylim(max(0, low - 6), min(high + 6, 127))
        elif pitch_range:
            pr = np.array(pitch_range)
            if pr.ndim == 1:
                ax.set_ylim(pr[0], pr[1])
            elif pr.ndim == 2:
                ax.set_ylim(pr[a][0], pr[a][1])
        ylim = ax.get_ylim()
                
        for bar_step in range(int(xlim[0]), int(xlim[1])+1, bar_res):
            ax.vlines(bar_step - 0.5, 0, 127)
            for beat in range(1, 4):
                ax.vlines(bar_step + beat_res * beat - 0.5, 0, 127, linestyles='dashed')

        for k, color in major:
            linewidth = 2.0 if k == 0 else 1.0
            for h in range(int(ylim[0]), int(ylim[1])):
                if h % 12 == k:
                    ax.hlines(h, xlim[0], xlim[1], linestyles='-', linewidth=linewidth, color=color)
    
    ppr.downbeat = downbeat
    
    rcParams['figure.figsize'] = orgSize


def soundfont():
    soundfont = ""
    pf = platform.system()
    # ubuntu
    if pf == 'Linux':
        soundfont = "../gsfont/gsfont.sf2"
    # mac
    if pf == 'Darwin':
        soundfont = "./data/GeneralUser_GS_v1.471.sf2"
    return soundfont

def pm_to_wave(pm, wave_file_name, sf_path, fs=44100):
    
    audio = pm.fluidsynth(fs, sf_path)
    
    # 16bit=2byte符号付き整数に変換してノーマライズ [-32768  ~ 32767]
    audio = np.array(audio * 32767.0, dtype="int16") # floatだと情報量が多くなる
    audio_stereo = np.c_[audio, audio] # ステレオ化
    spw.write(wave_file_name, fs, audio_stereo) # 書き出し
    
    return audio

def ppr_to_audio(ppr, save_dir, sfpath=soundfont(), tempo=120, save_npy=False, save_midi=True, convert_mp3=True):
    song_name = ppr.name
    wave_file_path = os.path.join(save_dir, f"{song_name}.wav")
    pm = ppr.to_pretty_midi(constant_tempo=tempo)
    audio = pm_to_wave(pm, wave_file_path, sfpath)

    print("wave file length:", len(audio))
    print("wave file saved to", wave_file_path)
    
    if save_npy:
        npy_path = os.path.join(save_dir, f'{song_name}.npy')
        np.save(npy_path, ppr)
        print(f"{song_name}.npy saved!")

    if save_midi:
        midi_path = os.path.join(save_dir, f'{song_name}.midi')
        ppr.write(midi_path)
        print(f"{song_name}.midi file saved!")
    
    if convert_mp3:
        sound = AS.from_wav(wave_file_path)
        mp3_file_path = f"{wave_file_path[:-4]}.mp3"
        sound.export(mp3_file_path, format="mp3")
        os.remove(wave_file_path)
        print("The wave file is replaced to", mp3_file_path, '\n')
    else:
        return Audio(wave_file_path)

    return Audio(mp3_file_path)
