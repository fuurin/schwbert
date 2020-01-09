# preparation/theorytab.pyで定義した各種モジュール
# embedding conditioningバージョン

import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from attrdict import AttrDict
from .save_and_load import save_model, load_model

class VecSeqEmbedding(nn.Module):
    def __init__(self, input_size, output_size, padding_idx=None):
        super(VecSeqEmbedding, self).__init__()
        self.pad_id = padding_idx
        self.embedding = nn.Linear(input_size, output_size, bias=False)
    
    def forward(self, input_seq):
        if self.pad_id is not None:
            input_seq[:, :, self.pad_id] = 0
        embedded = self.embedding(input_seq)
        return embedded
    
class ConditionalEmbeddings(nn.Module):
    def __init__(self, config, input_vocab_size, condition_vocab_size, input_pad_id=None):
        super(ConditionalEmbeddings, self).__init__()
        
        self.pad_id = input_pad_id
        self.emb_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.beat_res = config.beat_resolution
        self.bar_num = config.bar_num
        self.beats_in_bar = config.beats_in_bar
        self.bar_step_num = self.beats_in_bar * self.beat_res
        self.step_num = self.bar_step_num * self.bar_num
        
        # 各種埋め込み層
        self.input_embedding = VecSeqEmbedding(input_vocab_size, self.emb_size)
        self.condition_embedding = VecSeqEmbedding(condition_vocab_size, self.emb_size)
        self.step_embedding = nn.Embedding(self.bar_step_num, self.emb_size)
        self.beat_embedding = nn.Embedding(self.beats_in_bar, self.emb_size)
        self.bar_embedding = nn.Embedding(self.bar_num, self.emb_size)
        
        # 後処理用の層
        self.concat_dense = nn.Linear(self.emb_size*5, config.hidden_size)    
        self.post = nn.Sequential(
            LayerNorm(config.hidden_size, eps=1e-8),
            nn.Dropout(config.dropout_prob, inplace=True)
        )
        
        # 位置関係のID列は固定
        all_step_ids = torch.arange(self.step_num, dtype=torch.float32)
        self.step_ids = (all_step_ids % self.bar_step_num).type(torch.long)
        self.beat_ids = torch.floor((all_step_ids % self.bar_step_num) / self.beat_res).type(torch.long)
        self.bar_ids = torch.floor(all_step_ids / self.bar_step_num).type(torch.long)
        
    def concat_fixed_embeddings(self, input_emb, condition_emb):
        device = input_emb.device
        batch_size = len(input_emb)

        # 埋め込みベクトルを作成
        step_emb = self.step_embedding(self.step_ids.to(device)).repeat(batch_size, 1, 1)
        beat_emb = self.beat_embedding(self.beat_ids.to(device)).repeat(batch_size, 1, 1)
        bar_emb = self.bar_embedding(self.bar_ids.to(device)).repeat(batch_size, 1, 1)
        
        # 連結 & サイズ合わせ
        embeddings = torch.cat((input_emb, condition_emb, step_emb, beat_emb, bar_emb),-1)
        # embeddings = torch.cat((input_emb, condition_emb, step_emb),-1)
        embeddings = self.concat_dense(embeddings)
        
        return embeddings
    
    def forward(self, input_vecs, condition_vecs):
        input_emb = self.input_embedding(input_vecs)
        condition_emb = self.condition_embedding(condition_vecs)
        embeddings = self.concat_fixed_embeddings(input_emb, condition_emb)
        embeddings = self.post(embeddings)
        
        if self.pad_id is not None:
            is_not_pad = (input_vecs[:, :, self.pad_id] == 0).unsqueeze(-1)
            embeddings = embeddings * is_not_pad
            
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, dropout=False):
        super(BertSelfAttention, self).__init__()
        
        self.hidden_size = config.hidden_size # 24
        self.attention_head_num = config.attention_head_num # 4
        self.attention_head_size = self.hidden_size // self.attention_head_num # 6
        
        # Self-Attentionの特徴量を作成する全結合層
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.softmax = nn.Softmax(dim=-1)
        
        # bottleneck
        if dropout:
            self.dropout = nn.Dropout(config.dropout_prob, inplace=True)
        else:
            self.dropout = None
        
    def separate_into_heads(self, single):
        # multi-head attention用にテンソルの形を変換
        # [batch, steps, hidden] -> [batch, head_num, steps, head_size]
        multi_shape = single.size()[:-1] + (self.attention_head_num, self.attention_head_size)
        multi = single.view(*multi_shape).permute(0, 2, 1, 3)
        return multi
    
    def extend_pad(self, pad, paded_value=-1e9):
        if self.attention_head_num > 1:
            # multi-head attention用にpadの形を(batch, 1, 1, step_num)にする
            extended_pad = pad.unsqueeze(1).unsqueeze(2) # multi-headに次元を対応
        else:
            # single attention用にpadの形を(batch, 1, step_num)にする
            extended_pad = pad.unsqueeze(1)
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
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # multi-head Attentionとして分岐
        if self.attention_head_num > 1:
            query = self.separate_into_heads(query)
            key = self.separate_into_heads(key)
            value = self.separate_into_heads(value)
        
        # 特徴量同士の類似度を求める
        score = torch.matmul(query, key.transpose(-1, -2))
        score = score / math.sqrt(self.attention_head_size) # Scaled Dot-Product Attention
        
        # マスクをかける
        # 足し算なのは，attention_padに0か-infが入っているため
        # -infはsoftmax正規化したときに0になる
        score = score + self.extend_pad(pad)
        
        # AttentionMapの正規化とドロップアウト
        prob = self.softmax(score)
        if self.dropout is not None:
            prob = self.dropout(prob)
        
        # Attenton Mapをvalueに掛け算
        context = torch.matmul(prob, value)
        
        # multi-head Attentionの出力を結合
        if self.attention_head_num > 1:
            context = self.marge_heads(context)
        
        if get_probs:
            return context, prob
        else:
            return context



class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = LayerNorm(config.hidden_size, eps=1e-8)
        self.dropout = nn.Dropout(config.dropout_prob, inplace=True)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.attn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, input_pad, get_probs=False):
        if get_probs:
            output, probs = self.attn(input_tensor, input_pad, get_probs)
            output = self.output(output, input_tensor)
            return output, probs
        else:
            output = self.attn(input_tensor, input_pad, get_probs)
            output = self.output(output, input_tensor)
            return output



class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.intermediate = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.LeakyReLU(0.2) # WGAN-gpの勾配を使うところで実装がされていないとのことなので代用
        )
    
    def forward(self, hidden_states):
        return self.intermediate(hidden_states)

    
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
    
    def forward(self, hidden_states):
        hidden_states = self.intermediate(hidden_states)
        # return F.gelu(hidden_states) # WGAN-gpの勾配を使うところで実装がされていないとのことなので代用
        return F.leaky_relu(hidden_states, negative_slope=0.2, inplace=True)


class BertOutput(BertSelfOutput):
    def __init__(self, config):
        super(BertOutput, self).__init__(config)        
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)



class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attn = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    
    def forward(self, input_tensor, input_pad, get_probs=False):
        if get_probs:
            output, probs = self.attn(input_tensor, input_pad, get_probs)
            intermediate_output = self.intermediate(output)
            output = self.output(intermediate_output, output)
            return output, probs
        else:
            output = self.attn(input_tensor, input_pad, get_probs)
            intermediate_output = self.intermediate(output)
            output = self.output(intermediate_output, output)
            return output



class BertStack(nn.Module):
    def __init__(self, config):
        super(BertStack, self).__init__()
        self.layer_num = config.attention_layer_num    
        self.share_all_params = config.get("share_all_bert_params", False)
        if self.share_all_params:
            self.shared_attention = BertLayer(config)
            self.attention_list = range(self.layer_num)
        else:
            attention_list = [BertLayer(config) for _ in range(self.layer_num)]
            self.attention_list = nn.ModuleList(attention_list)
    
    def forward(self, input_tensor, input_pad, 
                get_all_outputs=False, get_probs=False):
        
        all_outputs, all_probs = [], []
        # Dの軽量版を作るとき，Attentionを使わない
        if self.attention_list is not None:
            for attention in self.attention_list:
                if self.share_all_params:
                    attention = self.shared_attention

                if get_probs:
                    input_tensor, probs = attention(
                        input_tensor, input_pad, get_probs=get_probs)
                else:
                    input_tensor = attention(
                        input_tensor, input_pad, get_probs=get_probs)

                # 12段すべての出力を見る場合
                if get_all_outputs:
                    all_outputs.append(input_tensor)
                    if get_probs:
                        all_probs.append(probs)
                
        # 最終段のAttentionのみ必要な場合
        if not get_all_outputs:
            all_outputs = input_tensor
            if get_probs:
                all_probs = probs
        
        if get_probs:
            return (all_outputs, all_probs)
        else:
            return all_outputs



class ConditionalBertBody(nn.Module):
    def __init__(self, config, embeddings):
        super(ConditionalBertBody, self).__init__()
        self.config = config
        self.input_pad_id = config.melody_pad_id
        self.condition_pad_id = config.chord_pad_id
        self.embeddings = embeddings
        self.bert_stack = BertStack(config)
    
    def make_pad(self, input_tensor, pad_id):
        pad = (input_tensor[:, :, pad_id] == 0)
        pad = pad.to(torch.float32).to(input_tensor.device)
        return pad
    
    def forward(self, input_tensor, condition_tensor, get_all_outputs=False, get_probs=False):
        embeddings = self.embeddings(input_tensor, condition_tensor)
        pad = self.make_pad(condition_tensor, self.condition_pad_id)
        stack_out = self.bert_stack(embeddings, pad, get_all_outputs, get_probs)
        return stack_out



def make_body(config):
    conditional_emb = ConditionalEmbeddings(config, 
                     config.melody_vocab_size,
                     config.chord_vocab_size,
                     config.melody_pad_id)
    body = ConditionalBertBody(config, conditional_emb)
    return body



def save_body(config, body, epoch_num, directory):
    emb_name = save_model(config, body.embeddings, epoch_num, directory)
    bert_stack_name = save_model(config, body.bert_stack, epoch_num, directory)
    return AttrDict({
        "embeddings": emb_name, 
        "bert_stack": bert_stack_name
    })

def load_body(config, directory):
    state_name_dict = config["state_names"]
    emb_name = state_name_dict['embeddings']
    bert_stack_name = state_name_dict['bert_stack']
    
    conditional_emb = ConditionalEmbeddings(config, 
                     config.melody_vocab_size, config.chord_vocab_size,
                     config.melody_pad_id)
    conditional_emb = load_model(conditional_emb, emb_name, directory)
    
    body = ConditionalBertBody(config, conditional_emb)
    body.bert_stack = load_model(body.bert_stack, bert_stack_name, directory)
    
    return body
