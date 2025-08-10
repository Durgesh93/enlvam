import torch
import torch.nn as nn
import torch.nn.functional as F

class DSNT(nn.Module):

    def __init__(self,T):
        super(DSNT, self).__init__()
        self.temp                      = T
        
    def forward(self, x):
        norm_heatmap                   = F.softmax(x.flatten(-2,-1)/self.temp,dim=-1).view(x.shape)
        
        # Get coordinate grids
        yy, xx                         = torch.meshgrid([torch.arange(norm_heatmap.shape[-2]), torch.arange(norm_heatmap.shape[-1])],indexing='ij')
        yy, xx                         = yy.float(), xx.float()
        yy                             = yy.to(norm_heatmap.device)
        xx                             = xx.to(norm_heatmap.device)
        
        # Multiply and sum
        yy_loc                         = torch.sum(norm_heatmap * yy, dim=[-2, -1]).view(norm_heatmap.shape[0], norm_heatmap.shape[1], 1)
        xx_loc                         = torch.sum(norm_heatmap * xx, dim=[-2, -1]).view(norm_heatmap.shape[0], norm_heatmap.shape[1], 1)
        coords                         = torch.cat([yy_loc, xx_loc], 2)
       
        return norm_heatmap,coords
      
    
