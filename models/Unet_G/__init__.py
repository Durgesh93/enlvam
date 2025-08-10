import torch.nn as nn
from ..core.unet import UNet
from ..core.dsnt import DSNT


class NNModule(nn.Module):

    def __init__(self, cfg):
        super(NNModule, self).__init__()
        self.cfg     = cfg
        self.unet    = UNet(**self.cfg.unet)
        self.dsnt    = DSNT(**self.cfg.dsnt)   
    
    def forward(self, x):
        inputs                    = x['inputs']
        x_out                     = self.unet(inputs)
        norm_heatmap,coords       = self.dsnt(x_out)
        x['pred_normheatmaps']    = norm_heatmap
        x['pred_coords']          = coords
        return x
