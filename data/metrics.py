import warnings
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class CatDictMetric(Metric):

    def __init__(self,cfg):
        super().__init__()
        self.cfg          = cfg
        for kp in self.cfg.params:
            self.add_state(kp,default=[],dist_reduce_fx='cat')
        
    def update(self, X,y):
        for k, v in X.items():
            if  f'x_{k}' in self.cfg.params:
                v = v.to(self.device)
                self.__dict__[f'x_{k}'].append(v)

        for k, v in y.items():
            if f'y_{k}' in self.cfg.params:
                v = v.to(self.device)
                self.__dict__[f'y_{k}'].append(v)
        
    def compute(self):
        X          =  {}
        y          =  {}
        for key  in self.cfg.params:
            metric       = self.__dict__[key]
            if len(metric):
                cat_arr      = dim_zero_cat(metric).detach().cpu()
                prefix,*name = key.split('_')
                name         = '_'.join(name)
                if prefix == 'x':
                    X[name] = cat_arr
                else:
                    y[name] = cat_arr
            else:
                warnings.warn(f"Metric {key} is empty")
        return X,y


class LVAM_Metrics_Base(CatDictMetric):

    def __init__(self, cfg):
        super().__init__(cfg)
    
    def compute_eval(self,X,y,paired=False,batch_reduction=False):
        gt_bcoords          = X['gt_bcoords']
        pred_bcoords        = X['pred_bcoords']
        gt_mask             = X['gt_valid']

        if 'pred_bregSL_projbcoords' not in X:
            X['pred_bregSL_projbcoords'] = X['pred_bcoords']

        proj_bcoords        = X['pred_bregSL_projbcoords']
        ratio               = y['bpix2cmratio']
        b                   = gt_bcoords.shape[0]

        prefix          = {'pix': 1,'mm':10*ratio,'cm':ratio}
        scores          = {'predicted':{},'projected':{}}

        for k in scores.keys():
            for pre in prefix.keys():
                rat                 = prefix[pre]
                gt                  = gt_bcoords*rat 
                pred                = pred_bcoords*rat
                proj                = proj_bcoords*rat 
                scores_k            = scores[k]
                
                if not paired:
                    pred_segments   = self.get_paired_format(pred) # bxN-1x2x2
                    proj_segments   = self.get_paired_format(proj)
                    gt_segments     = self.get_paired_format(gt)   # bxN-1x2x2
                    gt_mask_seg     = self.get_paired_format(gt_mask) # bxN-1x2
                else:
                    pred_segments   = pred.reshape(b,-1,2,2)
                    proj_segments   = proj.reshape(b,-1,2,2)
                    gt_segments     = gt.reshape(b,-1,2,2)
                    gt_mask_seg     = gt_mask.reshape(b,-1,2)
                

                
                N                   = pred_segments.shape[1]
                mask_coords         = (gt_mask_seg == 1).all(axis=-1)

                MAE             = np.zeros((b,N))
                MAPE            = np.zeros((b,N))
                MCE             = np.zeros((b,N))

                
                for idx in range(N):
                    gt_idx                             = gt_segments[mask_coords[:,idx],idx:idx+1]
                    pred_idx                           = pred_segments[mask_coords[:,idx],idx:idx+1]
                    proj_idx                           = proj_segments[mask_coords[:,idx],idx:idx+1]
                    if k == 'predicted':
                        MAE_idx                        = self.DE(gt_idx,pred_idx,abs=True,per=False)[:,0]
                        MAPE_idx                       = self.DE(gt_idx,pred_idx,abs=True,per=True)[:,0]
                        CE_idx                         = self.mean_CE(gt_idx,pred_idx)[:,0]
                    else:
                        MAE_idx                        = self.DE(gt_idx,proj_idx,abs=True,per=False)[:,0]
                        MAPE_idx                       = self.DE(gt_idx,proj_idx,abs=True,per=True)[:,0]
                        CE_idx                         = self.mean_CE(gt_idx,proj_idx)[:,0]
                    MAE[mask_coords[:,idx],idx]        = MAE_idx
                    MAPE[mask_coords[:,idx],idx]       = MAPE_idx
                    MCE[mask_coords[:,idx],idx]        = CE_idx

                scores_k[f'{pre}/MAE/lv']    = MAE.mean(axis=-1)
                scores_k[f'{pre}/MAPE/lv']   = MAPE.mean(axis=-1)
                scores_k[f'{pre}/MCE/lv']    = MCE.mean(axis=-1)
                
                if N == 3:
                    for idx,s in enumerate(['IVS','LVID','LVPW']):
                        scores_k[f'{pre}/MAE/{s}']     = MAE[:,idx]
                        scores_k[f'{pre}/MAPE/{s}']    = MAPE[:,idx]
                        scores_k[f'{pre}/MCE/{s}']     = MCE[:,idx]
                scores_k['dataid'] = y['dataid']
                
        if batch_reduction:
            for k in ['dataid']:
                scores['projected'].pop(k)
                scores['predicted'].pop(k)
            for k,v in scores['projected'].items():
                scores['projected'][k] = scores['projected'][k].mean(axis=0)
            for k,v in scores['predicted'].items():
                scores['predicted'][k] = scores['predicted'][k].mean(axis=0)
        return scores

    def plot_eval_bmode(self,X,y,processing):
        pass 

    def plot_eval_ammmode(self,X,y,processing):
        movieids             = np.unique(y['movieid'])
        pred_mcoords         = X['pred_mcoords']
        keyframe_idx         = y['keyframe_idx']
        gt_brefcoords        = X['gt_brefcoords']
        
        grid_size            = int(np.ceil(np.sqrt(len(movieids))))

        fig, axes = plt.subplots(
            grid_size, grid_size,
            figsize=(6*grid_size,2*grid_size)
        )

        for movieid, ax in zip(movieids, axes.flatten()):
            info       = processing.split.loc[processing.split['movieid']==movieid]
            dataids    = list(info['dataid'])
            movie      = processing.datasource.get_data(dataids[0])['bmovie']
            fidx       = y['movieid'] == movieid
            kfidx      = keyframe_idx[fidx]
            mcoords    = pred_mcoords[fidx]
            brefcoords = gt_brefcoords[fidx]

            
            inner_grid = gridspec.GridSpecFromSubplotSpec(
                            1, 2, 
                            subplot_spec=ax.get_subplotspec(), 
                            wspace=0, 
                            hspace=0
                        )
            
            for kf,refcoord,mcoord,grid in zip(kfidx,brefcoords,mcoords,inner_grid):
                inner_ax   = fig.add_subplot(grid)

                bclip,aidx = processing.echotools.clip_movie(
                                    movies=movie,
                                    Aidx=kf,
                                    WS=processing.cfg.window_size,
                                    batchmode=False
                                )
                            
                _,h,w,_    = bclip.shape
                vb         = processing.echotools.VBScanLine(
                                    reference = refcoord,
                                    Aidx=aidx, 
                                    S=0,
                                    R=0,
                                    image_dim=(h,w),
                                    batchmode=False
                                )
                brefSL     = vb.bSL()

                amm_patch  =  processing.echotools.clip2amm(
                                bclip,
                                brefSL,
                                batchmode=False
                            )
                inner_ax.imshow(amm_patch,cmap='gray')
                inner_ax.scatter(mcoord[:,1],mcoord[:,0],marker='x', color='orange', s=50)
                inner_ax.axis('off')
            
            ax.text(
                0.01, 0.98,
                f'MovID: {movieid}', 
                ha='left', va='top',
                transform=ax.transAxes,
                color='white',
                bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.3')
            )
            ax.axis('off')
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, 1:]
        plt.close(fig)
        return image
    
    def get_paired_format(self,coords):
        coord_pair      = np.stack((coords[:,:-1], coords[:,1:]), axis=2)
        return coord_pair
        
    def D(self,segment):
        # given N segments  
        fc_c         = segment[:,:,0,:]      #bxNx2
        sc_c         = segment[:,:,1,:]      #bxNx2
        d            = np.linalg.norm(fc_c-sc_c,axis=-1)  # bxN
        return d

    def D_btw(self,segment_c,segment_c_1):
        gt_midpoints     = segment_c.mean(axis=2)  # Shape (B, N, 2)
        pred_midpoints   = segment_c_1.mean(axis=2)  # Shape (B, N, 2)
        distances        = np.linalg.norm(gt_midpoints-pred_midpoints,axis=-1)
        return distances
    
    def A_btw(self,segment_c,segment_c1):
        gt_slopes      = (segment_c[:, :, 1, 0] - segment_c[:, :, 0, 0]) / (segment_c[:, :, 1, 1] - segment_c[:, :, 0, 1] + 1e-8)
        pred_slopes    = (segment_c1[:, :, 1, 0] - segment_c1[:, :, 0, 0]) / (segment_c1[:, :, 1, 1] - segment_c1[:, :, 0, 1] + 1e-8)
        angles_radians = np.arctan(np.abs((pred_slopes - gt_slopes) / (1 + pred_slopes * gt_slopes + 1e-8)))  # (B, N)
        angles_degrees = np.degrees(angles_radians)
        return angles_degrees[:,0]
        
    def DE(self,segment_c,segment_c1,abs=True,per=False):
        dsegment_c       = self.D(segment_c)
        dsegment_c1      = self.D(segment_c1)
        if abs:
            error        = np.abs((dsegment_c - dsegment_c1))
        else:
            error        = dsegment_c - dsegment_c1
        if per:
            eps          = np.finfo(np.float64).eps
            error        = (error+eps)/(dsegment_c+eps)       
        return error
        
    def mean_CE(self,segment,segment_c1):
        error            = np.linalg.norm(segment - segment_c1,axis=-1).mean(axis=-1)
        return error
    
