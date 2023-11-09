
from openprompt.data_utils import InputFeatures
import os
import torch
from torch import nn
from typing import *
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt.prompts import MixedTemplate

class PtuningTemplate(MixedTemplate):
    """
    Args:
        model (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        prompt_encoder_type (:obj:`str`): head above the embedding layer of new tokens. Can be ``lstm`` or ``mlp``.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """

    registered_inputflag_names = ["soft_token_ids", "loss_ids", 'shortenable_ids']

    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 pos_emd=None,
                 head_num=None,
                 pos_size=None,
                 text:  Optional[List[str]] = None,
                 prompt_encoder_type: str = "lstm",
                 args = None,
                ):
        super().__init__(model = model,
                         tokenizer=tokenizer,
                        )
        #self.args = args
        self.pos_emd = pos_emd
        self.head_num = head_num
        self.pos_size = pos_size
        self.args = args
        if head_num is not None:
            self.multihead_attention = SoftModel(query_size=args.query_size, num_att_layers=args.num_att_layers,input_dim=768, head_num=head_num,normal = args.normal, last_layer = args.last_layer)
            if pos_emd is not None:
                if args.one_hot:
                    self.pos_embedding = nn.Linear(in_features=pos_size, out_features=768)
                else:
                    self.pos_embedding =  nn.Embedding(pos_size, 768)

            if self.args.pe_pos:
                self.pe_pos = PositionalEncoding(d_model =768) 
        self.prompt_encoder_type = prompt_encoder_type
        self.text = text
        #self.args= args

    def on_text_set(self):
        r"""
        when template text was set, generate parameters needed in p-tuning input embedding phrase
        """
        super().on_text_set()
        self.num_soft_token = sum([soft_id != 0 for soft_id in self.soft_token_ids])
        self.generate_parameters()

    def generate_parameters(self) -> None:
        r"""
        generate parameters needed for new tokens' embedding in P-tuning
        """
        if self.num_soft_token == 0: return

        self.new_embedding = nn.Embedding(self.num_soft_token, self.embedding_size)
        self.new_ids = nn.Parameter(torch.LongTensor(list(range(self.num_soft_token))), requires_grad = False)
        if self.prompt_encoder_type == "lstm":
            self.new_lstm_head = nn.LSTM(
                input_size = self.embedding_size,
                hidden_size = self.embedding_size,
                num_layers = 2,
                bidirectional = True,
                batch_first = True
            )
            self.new_mlp_head = nn.Sequential(
                nn.Linear(2*self.embedding_size, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
        elif self.prompt_encoder_type == "mlp":
            self.new_mlp_head = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
        else:
            raise ValueError("unknown prompt_enocder_type")


    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        r"""
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for new tokens, use a brand new embedding layer, with MLP or LSTM head
        """
        inputs_embeds = self.raw_embedding(batch['input_ids'])
        softemd = None
        batch_softemd = None
        if self.num_soft_token != 0:
            #print('att')
            if self.head_num is not None:
                left_embeds = self.raw_embedding(batch['left_q_input_ids'])
                right_embeds = self.raw_embedding(batch['right_q_input_ids'])
                if self.pos_emd:
                    if self.args.one_hot:
                        batch['left_pos'] = torch.nn.functional.one_hot(batch['left_pos'], num_classes=self.pos_size).float()
                        batch['right_pos'] = torch.nn.functional.one_hot(batch['right_pos'], num_classes=self.pos_size).float()
                    left_embeds  += self.pos_embedding(batch['left_pos'])
                    right_embeds += self.pos_embedding(batch['right_pos'])
                if self.args.pe_pos:
                    left_embeds = self.pe_pos(left_embeds)
                    right_embeds = self.pe_pos(right_embeds)
                left_soft_embeds =  self.multihead_attention(left_embeds, batch['left_attention_mask'].bool())
                right_soft_embeds = self.multihead_attention(right_embeds, batch['right_attention_mask'].bool())
                soft_embeds = torch.concat([left_soft_embeds,right_soft_embeds],dim=1)
                batch_softemd = torch.concat([left_soft_embeds,right_soft_embeds],dim=0)
                #print(self.num_soft_token,self.head_num)
                assert self.num_soft_token == self.args.soft_token_num*2

                replace_idxs = torch.nonzero(batch['soft_token_ids'] > 0).view(-1, self.num_soft_token, 2)
                for b in range(replace_idxs.shape[0]):
                    for i in range(self.num_soft_token):
                        #print(replace_idxs[b][i][1])
                        inputs_embeds[b][replace_idxs[b][i][1]] = soft_embeds[b][i]
            else:
                new_embeds = self.new_embedding(self.new_ids).unsqueeze(0)
                if self.prompt_encoder_type == "lstm":
                    new_embeds = self.new_lstm_head(new_embeds)[0]
                new_embeds = self.new_mlp_head(new_embeds)

                replace_idxs = torch.nonzero(batch['soft_token_ids']>0).view(-1, self.num_soft_token, 2)
                for b in range(replace_idxs.shape[0]):
                    for i in range(self.num_soft_token):
                        inputs_embeds[b][replace_idxs[b][i][1]] = new_embeds[0][i]

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        return batch,batch_softemd


class SoftModel(nn.Module):
    def __init__(self, num_att_layers,input_dim, head_num, normal, last_layer,query_size):
        super(SoftModel, self).__init__()
        self.layers = nn.ModuleList([ MultiheadAttention(input_dim, head_num, normal, last_layer,query_size) for _ in range(num_att_layers) ])
        self.num_att_layers = num_att_layers

    def forward(self, inputs, attention_mask=None):

        for i,m in enumerate(self.layers):
            inputs = m(inputs, attention_mask, i == self.num_att_layers-1)

        return inputs



class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, head_num,normal , last_layer,query_size):
        super(MultiheadAttention, self).__init__()
        self.input_dim = input_dim
        self.head_num = head_num
        self.normal = normal
        self.last_layer = last_layer

        self.query_linear = nn.Linear(input_dim, head_num * input_dim)
        self.key_linear = nn.Linear(input_dim, head_num * input_dim)
        self.value_linear = nn.Linear(input_dim, head_num * input_dim)
        self. query_size = query_size
        if self.query_size :
            self.query_emd = nn.Embedding(query_size, 768)

        if self.last_layer:
            self.output_linear = nn.Linear(input_dim, input_dim)
        if self.normal:
            self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, inputs,attention_mask=None, last = False):
        batch_size, seq_length, _ = inputs.size()

        # Linear projections for queries, keys, and values
        if last and self.query_size:
            queries = self.query_linear(self.query_emd(torch.arange(0, self.query_size,device='cuda').repeat(batch_size, 1)))
        else:
            queries = self.query_linear(inputs)
        keys = self.key_linear(inputs)
        values = self.value_linear(inputs)

        # Split queries, keys, and values into multiple heads
        queries = queries.view(batch_size, -1, self.head_num, self.input_dim)
        keys = keys.view(batch_size, seq_length, self.head_num, self.input_dim)
        values = values.view(batch_size, seq_length, self.head_num, self.input_dim)

        # Transpose for matrix multiplication
        queries = queries.transpose(1, 2) # (batch_size, head_num, seq_length, input_dim)
        keys = keys.transpose(1, 2) # (batch_size, head_num, seq_length, input_dim)
        values = values.transpose(1, 2) # (batch_size, head_num, seq_length, input_dim)

        # Scaled Dot-Product Attention
        scaled_dot_product = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.input_dim).float())
        if attention_mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attention_weights = torch.softmax(scaled_dot_product, dim=-1)
        attention_output = torch.matmul(attention_weights, values)

        # Merge attention outputs of different heads
        attention_output = attention_output.transpose(1, 2).contiguous() # (batch_size, seq_length, head_num, input_dim)
        if self.query_size or not last:
            attention_output = attention_output.mean(dim=2)  # Take average over the sequence dimension
            attention_output = attention_output.view(batch_size, -1, self.input_dim)  # Reshape
        else:
            attention_output = attention_output.mean(dim=1)  # Take average over the sequence dimension
            attention_output = attention_output.view(batch_size, -1 , self.input_dim)  # Reshape
        if self.last_layer:
            attention_output= self.output_linear(attention_output)
        if self.normal:
            attention_output =  self.layer_norm(attention_output)
        return  attention_output
    
import torch
import math
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)#(max-len,1,d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        #x = x + self.pe[:x.size(1), :]
        return x
