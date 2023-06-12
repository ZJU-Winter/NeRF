import numpy as np
import torch
# some misc helper functions

# return 8 byte data ranging from (0-255)
def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)

# return mean square error
def img2mse(x, y): return torch.mean((x - y) ** 2)

# return psnr
def mse2psnr(x): return -10.*torch.log(x)/ torch.log(torch.Tensor([10.]))