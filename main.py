import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class ReversibleSelfAttention(nn.Module):


    def __init__(self, d_model : int, n_heads : int, dropout : float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)


        # storing the intermediate states for the reversibility
        # For this I am thinking to just for the list kind of thing

        self.stored_states = {}

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                store_states: bool = True, step_id: str = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        if store_states and step_id:
            self.stored_states[step_id] = {
                'input': x.clone(),
                'pre_norm': x.clone()
            }
        

        normed_x = self.layer_norm(x)
        

        q = self.w_q(normed_x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(normed_x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(normed_x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.w_o(attn_output)
        
        if store_states and step_id:
            self.stored_states[step_id].update({
                'attn_weights': attn_weights.clone(),
                'attn_output': attn_output.clone(),
                'output': output.clone()
            })
        
        return x + output


    def reverse(self, y : torch.Tensor, step_id : str) -> torch.Tensor:
        

        if step_id not in self.stored_states:
            raise ValueError(f"No stored state found for step_id : {step_id}")


        states = self.stored_states[step_id]


        x = y - states['output']

        # clearing up the saved states

        del self.stored_states[step_id]


        return x

class ReversibleFeedForward(nn.Module):


    "feed forward layer that maintains reversibility"

    def __init__(self, d_model : int, d_ff : int, dropout : float = 0.1):
        super().__init__()


        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)


        self.stored_states = {}


    def forward(self, x : torch.Tensor, store_states : bool = True, step_id : str = None) -> torch.Tensor:
        if store_states and step_id :
            self.stored_states[step_id] = {
                'input' : x.clone(),
                'pre_nrom' : x.clone()
            }


        normed_x = self.layer_norm(x)


        ff_output = self.linear2(self.dropout(F.relu(self.linear1(normed_x))))

        if store_states and step_id:
            self.stored_states[step_id]['ffn_output'] = ff_output.clone()



        return x + ff_output


    def reverse(self, y : torch.Tensor, step_id : str) -> torch.Tensor:
        

        if step_id not in self.stored_states:
            raise ValueError(f"No stored state found for step_id : {step_id}")


        states = self.stored_states[step_id]


        # reversing the residual connection

        x = y - states['ffn_output']

        del self.stored_states[step_id]

        return x


class ReversibleDecoderLayer(nn.Module):

    # a single decoder layer

    def __init__(self, d_model : int, n_heads : int, d_ff : int, dropout : float = 0.1):
        super().__init__()


        self.self_attn = ReversibleSelfAttention(d_model, n_heads, dropout)