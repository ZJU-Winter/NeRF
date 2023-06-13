import torch.nn as nn
import torch 
from typing import Callable

def get_embedder(multires, i=0):
    """
    positional encoding (embedding)
    @parameter multires: maximum frequency (after log2)
    @parameter i: default 0, if set to -1, return identity
    @return embed: embedding function
    @return out_dim: output dimmension
    """
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    input_dims = 3
    max_freq_log2 = multires - 1
    num_freqs = multires
    periodic_fns = [torch.sin, torch.cos]
    out_dim = 0
    embed_fns = []

    # embed functions including input itself
    embed_fns.append(lambda x : x)
    out_dim += input_dims

    # adopt log sampling by default
    # $f(p) = (sin(2^0 \pi p), cos(2^1 \pi p), ..., sin(2^(L-1) \pi p), sin(2^(L-1) \pi p))$
    # freq_bands = 2^0, 2^1, ...2^(L-1)
    freq_bands = 2.**torch.linspace(0., max_freq_log2, steps=num_freqs)

    for freq in freq_bands:
        for p_fn in periodic_fns:
            embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
            out_dim += input_dims 
    
    def embed(x):
        return torch.cat([fn(x) for fn in embed_fns], -1)
    
    return embed, out_dim