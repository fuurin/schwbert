# preparation/theorytab.pyで定義したモジュールの保存と読み込みに使う関数

import os, json, datetime
import torch
from attrdict import AttrDict
from .multi_gpu import MultiGPUWrapper

def make_state_name(config, model, epoch_num):
    if issubclass(model.__class__, MultiGPUWrapper):
        model_name = model.module.__class__.__name__
    else:
        model_name = model.__class__.__name__
    
    nickname = config.nickname
    E = epoch_num
    H = config.hidden_size
    I = config.intermediate_size
    A = config.attention_layer_num
    AH = config.attention_head_num
    
    state_name = f"{nickname}:{model_name}:E={E}H={H}I={I}A={A}AH={AH}"
    
    if hasattr(model, 'condition_str'):
        state_name += model.condition_str
    
    return state_name

def save_model(config, model, epoch_num, directory):
    if issubclass(model.__class__, MultiGPUWrapper):
        model = model.module
    
    state_name = f"{make_state_name(config, model, epoch_num)}.pth"
    state_path = os.path.join(directory, state_name)
    torch.save(model.state_dict(), state_path)
    print(f"{state_name} saved")
    return state_name

def load_model(model, state_name, directory):
    file_path = os.path.join(directory, state_name)
    state_dict = torch.load(file_path)
    model.load_state_dict(state_dict)
    return model

def save_config(config, directory, state_names_dict={}):
    delta = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(delta).strftime('%Y-%m-%d_%H:%M')
    file_name = f"{config.nickname}:config@{now}.json"
    file_path = os.path.join(directory, file_name)
    config['state_names'] = dict(state_names_dict)
    json.dump(dict(config), 
              open(file_path, "w"), 
              ensure_ascii=False, 
              indent=4, 
              separators=(',', ': '))
    print(f"{file_name} saved")
    return file_name

def load_config(config_name, directory):
    file_path = os.path.join(directory, config_name)
    config = AttrDict(json.load(open(file_path, "r")))
    return config