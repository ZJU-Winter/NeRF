import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False) -> None:
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.relu = nn.ReLU()

        # 8 linear layers
        self.pts_linears = nn.Sequential(
            nn.Linear(input_ch, W),
            *[nn.Linear(W, W) if i not in skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )

        # 256 + view -> 128
        self.views_linears = nn.Sequential(nn.Linear(input_ch_views + W, W//2))

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
    
    def forward(self, input):
        input_pts, input_views = torch.split(input, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        
        # get view output
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            h = self.views_linears[0](h)
            h = self.relu(h)
            rgb = self.rgb_linear(h)
            return torch.cat([rgb, alpha], -1)
        else:
            return self.output_linear(h)