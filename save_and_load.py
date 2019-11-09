# preparation/theorytab.pyで定義したモジュールの保存と読み込みに使う関数

import os, json, datetime
import torch
from attrdict import AttrDict

def make_state_name(config, model, epoch_num):
    nickname = config.nickname
    model_name = model.__class__.__name__
    E = epoch_num
    H = config.hidden_size
    I = config.intermediate_size
    A = config.attention_layer_num
    AH = config.attention_head_num
    
    state_name = f"{nickname}:{model_name}:E={E}H={H}I={I}A={A}AH={AH}"
    
    if issubclass(model.__class__, PreTrainingModel):
        state_name += model.condition_str
    
    return state_name

def save_model(config, model, epoch_num, directory):
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

def save_body(config, body, epoch_num, directory):
    input_emb_name = save_model(config, body.input_embeddings, epoch_num, output_dir)
    condition_emb_name = save_model(config, body.condition_embeddings, epoch_num, output_dir)
    bert_stack_name = save_model(config, body.conditional_bert_stack, epoch_num, output_dir)
    return AttrDict({
        "input_embeddings": input_emb_name, 
        "condition_embeddings": condition_emb_name, 
        "conditional_bert_stack": bert_stack_name
    })

def load_body(config, input_emb, condition_emb, directory):
    state_name_dict = config["state_names"]
    
    input_emb_name = state_name_dict['input_embeddings']
    condition_emb_name = state_name_dict['condition_embeddings']
    bert_stack_name = state_name_dict['conditional_bert_stack']
    
    input_emb = load_model(input_emb, input_emb_name, directory)
    condition_emb = load_model(condition_emb, condition_emb_name, directory)
    body = ConditionalBertBody(config, melody_emb, chord_emb, 
                               config.melody_pad_id, config.chord_pad_id)
    body.conditional_bert_stack = load_model(body.conditional_bert_stack, 
                                             bert_stack_name, directory)
    
    return body

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
              sort_keys=True, 
              separators=(',', ': '))
    print(f"{file_name} saved")
    return file_name

def load_config(config_name, directory):
    file_path = os.path.join(directory, config_name)
    config = AttrDict(json.load(open(file_path, "r")))
    return config