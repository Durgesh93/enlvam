import torch
import torch.nn as nn


class L2_loss(nn.Module):
    def __init__(self,W):
        super().__init__()
        self.W               = W

    def forward(self,X,paired=False):
        batch_out     = {}
        pred          = X['pred_coords']
        target        = X['gt_coords']
        gt_masks      = X['gt_valid']
        pred          = pred.float()
        target        = target.float()
        mask          = (gt_masks ==1).all(dim=-1)
        target        = target[mask]
        pred          = pred[mask]
        loss          = torch.norm(target.flatten(-2,-1) - pred.flatten(-2,-1),dim=-1).mean()
        batch_out['L2_loss'] = loss.detach().clone()
        return loss,batch_out