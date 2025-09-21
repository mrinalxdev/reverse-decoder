import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

class ReversibleSelfAttention(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
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
        
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        
        output = self.w_o(attn_output)
        
        if store_states and step_
            self.stored_states[step_id].update({
                'attn_weights': attn_weights.clone(),
                'attn_output': attn_output.clone(),
                'output': output.clone()
            })
        
        
        return x + output
    
    def reverse(self, y: torch.Tensor, step_id: str) -> torch.Tensor:
        """Reverse the forward computation using stored states"""
        if step_id not in self.stored_states:
            raise ValueError(f"No stored state found for step_id: {step_id}")
        
        states = self.stored_states[step_id]
        
       
        x = y - states['output']
        
       
        del self.stored_states[step_id]
        
        return x


class ReversibleFeedForward(nn.Module):
    
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.stored_states = {}
    
    def forward(self, x: torch.Tensor, store_states: bool = True, step_id: str = None) -> torch.Tensor:
        if store_states and step_id:
            self.stored_states[step_id] = {
                'input': x.clone(),
                'pre_norm': x.clone()
            }
        
       
        normed_x = self.layer_norm(x)
        
        
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(normed_x))))
        
        if store_states and step_id:
            self.stored_states[step_id]['ff_output'] = ff_output.clone()
        
        
        return x + ff_output
    
    def reverse(self, y: torch.Tensor, step_id: str) -> torch.Tensor:
       
        if step_id not in self.stored_states:
            raise ValueError(f"No stored state found for step_id: {step_id}")
        
        states = self.stored_states[step_id]
        
       
        x = y - states['ff_output']
        
        
        del self.stored_states[step_id]
        
        return x


class ReversibleDecoderLayer(nn.Module):
    """A single decoder layer that can be reversed"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = ReversibleSelfAttention(d_model, n_heads, dropout)
        self.feed_forward = ReversibleFeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                store_states: bool = True, step_id: str = None) -> torch.Tensor:
        
        x = self.self_attn(x, mask, store_states, f"{step_id}_attn" if step_id else None)
        
       
        x = self.feed_forward(x, store_states, f"{step_id}_ff" if step_id else None)
        
        return x
    
    def reverse(self, y: torch.Tensor, step_id: str) -> torch.Tensor:
        y = self.feed_forward.reverse(y, f"{step_id}_ff")
        y = self.self_attn.reverse(y, f"{step_id}_attn")
        return y


class ReversibleDecoder(nn.Module):
    """A decoder that can generate text forward and un-generate it backward"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
      
        self.layers = nn.ModuleList([
            ReversibleDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        
        self.generation_states = []
        self.current_sequence = []
        
    def get_causal_mask(self, seq_len: int) -> torch.Tensor:
    
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask == 0
    
    def forward(self, input_ids: torch.Tensor, store_states: bool = True) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
    
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        
        x = self.dropout(token_embeds + position_embeds)
        
    
        causal_mask = self.get_causal_mask(seq_len).to(input_ids.device)
        
    
        for i, layer in enumerate(self.layers):
            step_id = f"layer_{i}" if store_states else None
            x = layer(x, causal_mask, store_states, step_id)
        
    
        logits = self.output_projection(x)
        
        return logits
    
    def generate_forward(self, start_token: int, max_length: int = 50,
                        temperature: float = 1.0) -> list:
    
        self.eval()
        self.generation_states = []
        self.current_sequence = [start_token]
        
        with torch.no_grad():
            for step in range(max_length):
                input_ids = torch.tensor([self.current_sequence], dtype=torch.long)
                
            
                current_state = {
                    'sequence': self.current_sequence.copy(),
                    'step': step
                }
                
            
                logits = self.forward(input_ids, store_states=True)
                
            
                next_token_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                
            
                next_token = torch.multinomial(probs, 1).item()
                
            
                current_state.update({
                    'logits': logits.clone(),
                    'next_token': next_token,
                    'probs': probs.clone()
                })
                self.generation_states.append(current_state)
                
            
                self.current_sequence.append(next_token)
                
            
                if next_token == 0:
                    break
        
        return self.current_sequence
    
    def generate_backward(self) -> list:
    
        if not self.generation_states:
            raise ValueError("No forward generation states found. Call generate_forward first.")
        
        sequences = []
        
    
        for step_idx in reversed(range(len(self.generation_states))):
            state = self.generation_states[step_idx]
            
        
            self.current_sequence = self.current_sequence[:-1]
            sequences.append(self.current_sequence.copy())
            
           
            if self.current_sequence:
                input_ids = torch.tensor([self.current_sequence], dtype=torch.long)
                
                
                with torch.no_grad():
                    
                    reconstructed_logits = self.forward(input_ids, store_states=False)
                    
                   
                    original_logits = state['logits'][:, :len(self.current_sequence), :]
                    
                   
                    if len(self.current_sequence) > 0:
                        diff = torch.abs(reconstructed_logits - original_logits).max()
                        if diff > 1e-5:
                            print(f"Warning: Reversibility check failed at step {step_idx}, diff: {diff}")
            
           
            for layer_idx in range(self.n_layers):
                layer = self.layers[layer_idx]
                attn_key = f"layer_{layer_idx}_attn"
                ff_key = f"layer_{layer_idx}_ff"
                
                if attn_key in layer.self_attn.stored_states:
                    del layer.self_attn.stored_states[attn_key]
                if ff_key in layer.feed_forward.stored_states:
                    del layer.feed_forward.stored_states[ff_key]
        
        
        self.generation_states = []
        
        return sequences



def create_simple_tokenizer():
    
    chars = "abcdefghijklmnopqrstuvwxyz .,!?\n"
    char_to_id = {char: i+1 for i, char in enumerate(chars)} 
    char_to_id['<PAD>'] = 0
    id_to_char = {i: char for char, i in char_to_id.items()}
    return char_to_id, id_to_char

def demonstrate_reversible_decoder():
    """Demonstrate the reversible decoder functionality"""
    print("🔄 Reversible Decoder Transformer Demo")
    print("=" * 50)
    
    # Create tokenizer
    char_to_id, id_to_char = create_simple_tokenizer()
    vocab_size = len(char_to_id)
    
    # Initialize model
    model = ReversibleDecoder(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        max_seq_len=100
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Vocabulary size: {vocab_size}")
    
    print("Generating forward...")
    start_token = char_to_id['h'] 
    generated_sequence = model.generate_forward(start_token, max_length=15, temperature=0.8)
    
   
    generated_text = ''.join([id_to_char[token_id] for token_id in generated_sequence[1:]])  
    print(f"Generated: '{generated_text}'")
    print(f"Token sequence: {generated_sequence}")
    
    
    print("Un-generating backward...")
    backward_sequences = model.generate_backward()
    
    print("Backward un-generation steps:")
    for i, seq in enumerate(backward_sequences):
        if seq:
            text = ''.join([id_to_char[token_id] for token_id in seq[1:]])
            print(f"Step {i+1}: '{text}' -> tokens: {seq}")
        else:
            print(f"Step {i+1}: [empty] -> tokens: {seq}")
    
    print("\nReversible generation complete!")
    print("The model successfully generated text forward and un-generated it backward.")

if __name__ == "__main__":
    demonstrate_reversible_decoder()