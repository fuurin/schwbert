# preparation/theorytab.pyで定義した各種モジュール
# attention conditioningバージョン

import math, random
import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
from attrdict import AttrDict
from .save_and_load import save_model, load_model

class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, fact_size, hidden_size, **kwargs):
        super(FactorizedEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, fact_size, **kwargs)
        self.out = nn.Linear(fact_size, hidden_size, bias=False)
    
    def forward(self, inputs):
        output = self.embedding(inputs)
        output = self.out(output)
        return output

class MusicEmbeddings(nn.Module):
    def __init__(self, config, vocab_size=None, pad_id=0):
        super(MusicEmbeddings, self).__init__()
        
        # EmbeddingLayer: (入力の種類数, 組込ベクトルサイズ)
        # 48とかのIDを受け取ってone-hot vectorと解釈して組込ベクトルへ変換する
        self.pad_id = pad_id
        self.fact_size = config.fact_size
        self.hidden_size = config.hidden_size
        self.beat_res = config.beat_resolution
        self.step_num = config.step_num
        self.beat_num = config.beats_in_bar * config.bar_num
        self.bar_num = config.bar_num
        self.bar_step_num = config.beats_in_bar * self.beat_res
        
        if vocab_size is not None:
            self.input_embedding = FactorizedEmbedding(
                vocab_size,       # 67 or 4097
                self.fact_size,   # 12
                self.hidden_size, # 24
                padding_idx=pad_id
            )
        
        self.step_embedding = FactorizedEmbedding(
            self.step_num,   # 768
            self.fact_size,  # 12
            self.hidden_size # 24
        )
        
        self.beat_embedding = FactorizedEmbedding(
            self.beat_num,   # 64
            self.fact_size,  # 12
            self.hidden_size # 24
        )
        
        self.bar_embedding = FactorizedEmbedding(
            self.bar_num,    # 16
            self.fact_size,  # 12
            self.hidden_size # 24
        )
        
        self.norm = LayerNorm(config.hidden_size, eps=1e-8)
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def add_fixed_embeddings(self, input_emb):
        # step ID -> ステップ埋め込みベクトル
        step_ids = torch.arange(self.step_num, dtype=torch.float32) # 0~767
        step_ids = step_ids.unsqueeze(0).expand(input_emb.shape[:-1]) # バッチ用の次元を追加
        step_ids = step_ids.to(input_emb.device)
        step_emb = self.step_embedding(step_ids.type(torch.long))
        
        # beat ID -> 拍埋め込みベクトル
        # 同じ数がbeat_res個続くようにする
        beat_ids = torch.floor(step_ids.clone() / self.beat_res)
        beat_emb = self.beat_embedding(beat_ids.type(torch.long))
        
        # bar ID -> 小節埋め込みベクトル
        # 同じ数がbar_step_num個続くようにする
        bar_ids = torch.floor(step_ids.clone() / self.bar_step_num)
        bar_emb = self.bar_embedding(bar_ids.type(torch.long))
        
        # 4つの埋め込みベクトルを足し合わせる
        # (batch_size, step_num, hidden_size)
        embeddings = input_emb + step_emb + beat_emb + bar_emb
        
        return embeddings
    
    def post_layers(self, tensor):
        tensor = self.norm(tensor)
        tensor = self.dropout(tensor)
        return tensor
    
    def forward(self, input_ids):
        # input_ids: (batch_size, step_num)の文章中の単語ID列
        
        # input ID -> 入力埋め込みベクトル
        input_emb = self.input_embedding(input_ids)
        
        # ステップ，拍，小節の埋め込みベクトルを加算
        embeddings = self.add_fixed_embeddings(input_emb)
                
        # 埋め込みベクトルを正規化 & Dropout
        embeddings = self.post_layers(embeddings)
        
        return embeddings


class MelodyEmbeddings(MusicEmbeddings):
    def __init__(self, config):
        super(MelodyEmbeddings, self).__init__(config, config.melody_vocab_size, config.melody_pad_id)


class ChordEmbeddings(MusicEmbeddings):
    def __init__(self, config):
        super(ChordEmbeddings, self).__init__(config, config.chord_vocab_size, config.chord_pad_id)




class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        
        self.hidden_size = config.hidden_size # 24
        self.attention_head_num = config.attention_head_num # 4
        self.attention_head_size = self.hidden_size // self.attention_head_num # 6
        
        # Self-Attentionの特徴量を作成する全結合層
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def separate_into_heads(self, single):
        # multi-head attention用にテンソルの形を変換
        # [batch, steps, hidden] -> [batch, head_num, steps, head_size]
        multi_shape = single.size()[:-1] + (self.attention_head_num, self.attention_head_size)
        multi = single.view(*multi_shape).permute(0, 2, 1, 3)
        return multi
    
    def extend_pad(self, pad, paded_value=-10000.0):
        # multi-head attention用にpadの形を(batch, 1, 1, step_num)にする
        extended_pad = pad.unsqueeze(1).unsqueeze(2) # multi-headに次元を対応
        extended_pad = (1.0 - extended_pad) * paded_value
        extended_pad = extended_pad.to(dtype=torch.float32)
        return extended_pad
    
    def marge_heads(self, multi):
        # multi-head attentionに分離したテンソルの形を元に戻す
        # [batch, head_num, steps, head_size] -> [batch, steps, hidden]
        multi = multi.permute(0, 2, 1, 3).contiguous()
        single_shape = multi.size()[:-2] + (self.hidden_size,)
        single = multi.view(*single_shape)
        return single
    
    def forward(self, hidden_states, pad, get_probs=False):
        
        # 入力を全結合層で特徴量変換(分岐前)
        marged_query = self.query(hidden_states)
        marged_key = self.key(hidden_states)
        marged_value = self.value(hidden_states)
        
        # multi-head Attentionとして分岐
        queries = self.separate_into_heads(marged_query)
        keyes = self.separate_into_heads(marged_key)
        values = self.separate_into_heads(marged_value)
        
        # 特徴量同士の類似度を求める
        scores = torch.matmul(queries, keyes.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size) # Scaled Dot-Product Attention
        
        # マスクをかける
        # 足し算なのは，attention_padに0か-infが入っているため
        # -infはsoftmax正規化したときに0になる
        scores = scores + self.extend_pad(pad)
        
        # AttentionMapの正規化とドロップアウト
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        
        # Attenton Mapをvalueに掛け算
        contexts = torch.matmul(probs, values)
        
        # multi-head Attentionの出力を結合
        context = self.marge_heads(contexts)
        
        if get_probs:
            return context, probs
        else:
            return context


class BertSelfConditioning(nn.Module):
    def __init__(self, config):
        super(BertSelfConditioning, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = LayerNorm(config.hidden_size, eps=1e-8)
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def forward(self, hidden_states, condition_tensor):
        conditioned_states = self.dense(hidden_states + condition_tensor)
        conditioned_states = self.dropout(conditioned_states)
        conditioned_states = self.norm(conditioned_states)
        return conditioned_states


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = LayerNorm(config.hidden_size, eps=1e-8)
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states + input_tensor)
        return hidden_states


class ConditionalBertAttention(nn.Module):
    def __init__(self, config):
        super(ConditionalBertAttention, self).__init__()
        self.condition_attn = BertSelfAttention(config)
        self.conditioning = BertSelfConditioning(config)
        self.attn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, condition_tensor, input_pad, condition_pad, get_probs=False):
        
        if get_probs:
            condition, condition_probs = self.condition_attn(condition_tensor, condition_pad, get_probs)
            conditioned_tensor = self.conditioning(input_tensor, condition)
            output, probs = self.attn(conditioned_tensor, input_pad, get_probs)
            output = self.output(output, input_tensor) # or conditioned_tensor?
            return output, probs, condition_probs
        else:
            condition = self.condition_attn(condition_tensor, condition_pad, get_probs)
            conditioned_tensor = self.conditioning(input_tensor, condition)
            output = self.attn(conditioned_tensor, input_pad, get_probs)
            output = self.output(output, input_tensor) # or conditioned_tensor?
            return output




def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(BertSelfOutput):
    def __init__(self, config):
        super(BertOutput, self).__init__(config)        
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)




class ConditionalBertLayer(nn.Module):
    def __init__(self, config):
        super(ConditionalBertLayer, self).__init__()
        self.attn = ConditionalBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    
    def forward(self, input_tensor, condition_tensor, input_pad, condition_pad, get_probs=False):
        if get_probs:
            output, probs, c_probs = self.attn(input_tensor, condition_tensor, input_pad, condition_pad, get_probs)
            intermediate_output = self.intermediate(output)
            output = self.output(intermediate_output, output)
            return output, probs, c_probs
        else:
            output = self.attn(input_tensor, condition_tensor, input_pad, condition_pad, get_probs)
            intermediate_output = self.intermediate(output)
            output = self.output(intermediate_output, output)
            return output


class ConditionalBertStack(nn.Module):
    def __init__(self, config):
        super(ConditionalBertStack, self).__init__()
        self.layer_num = config.attention_layer_num    
        self.share_all_params = config.get("share_all_bert_params", False)
        if self.share_all_params:
            self.shared_attention = ConditionalBertLayer(config)
            self.attention_list = range(self.layer_num)
        else:
            attention_list = [ConditionalBertLayer(config) for _ in range(self.layer_num)]
            self.attention_list = nn.ModuleList(attention_list)
    
    def forward(self, input_tensor, condition_tensor, input_pad, condition_pad, 
                get_all_outputs=False, get_probs=False):
        
        all_outputs, all_probs, all_c_probs = [], [], []
        
        for attention in self.attention_list:
            if self.share_all_params:
                attention = self.shared_attention
                
            if get_probs:
                input_tensor, probs, c_probs = attention(
                    input_tensor, condition_tensor, 
                    input_pad, condition_pad, 
                    get_probs=get_probs)
            else:
                input_tensor = attention(
                    input_tensor, condition_tensor, 
                    input_pad, condition_pad, 
                    get_probs=get_probs)
                
            # 12段すべての出力を見る場合
            if get_all_outputs:
                all_outputs.append(input_tensor)
                if get_probs:
                    all_probs.append(probs)
                    all_c_probs.append(c_probs)
                
        # 最終段のAttentionのみ必要な場合
        if not get_all_outputs:
            all_outputs = input_tensor
            if get_probs:
                all_probs = probs
                all_c_probs = c_probs
        
        if get_probs:
            return (all_outputs, all_probs, all_c_probs)
        else:
            return all_outputs




class ConditionalBertBody(nn.Module):
    def __init__(self, config, input_embeddings, condition_embeddings):
        super(ConditionalBertBody, self).__init__()
        self.config = config
        self.input_pad_id = input_embeddings.pad_id
        self.condition_pad_id = condition_embeddings.pad_id
        self.input_embeddings = input_embeddings
        self.condition_embeddings = condition_embeddings
        self.conditional_bert_stack = ConditionalBertStack(config)
    
    def make_pad(self, input_tensor, pad_id):
        if input_tensor.dim() == 2:
            pad = (input_tensor != pad_id)
        elif input_tensor.dim() == 3:
            pad = (input_tensor[:, :, pad_id] == 1)
        
        pad = pad.to(torch.float32).to(input_tensor.device)
        return pad
    
    def forward(self, input_tensor, condition_tensor, get_all_outputs=False, get_probs=False):
        input_pad = self.make_pad(input_tensor, self.input_pad_id)
        condition_pad = self.make_pad(condition_tensor, self.condition_pad_id)
        input_tensor = self.input_embeddings(input_tensor)
        condition_tensor = self.condition_embeddings(condition_tensor)
        stack_out = self.conditional_bert_stack(
                        input_tensor, condition_tensor,
                        input_pad, condition_pad,
                        get_all_outputs, get_probs)
        return stack_out
    


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
    body = ConditionalBertBody(config, input_emb, condition_emb)
    body.conditional_bert_stack = load_model(body.conditional_bert_stack, 
                                             bert_stack_name, directory)
    
    return body