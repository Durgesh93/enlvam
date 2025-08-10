import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

class BaseProcessing(object):

    def __init__(self,datasource,transform,cfg=None):
        self.cfg           = cfg
        self.datasource    = datasource
        self.transform     = transform
        self.encoders      = self.datasource.encoders
        self.split         = self.create_split(self.datasource.split)
        self.echotools     = self.cfg.echotools.get_instance()
    
    def create_split(self,split):
        return split

    def to_tensor(self, X, y, batchmode=False):
        X_tensors = {}
        y_tensors = {}
        if not batchmode:
            for k, v in X.items():
                X[k] = np.array([v])
            for k, v in y.items():
                y[k] = np.array([v])

        for k, v in X.items():
            if isinstance(v,list):
                X_tensors[k] = torch.from_numpy(np.array(v))
            elif isinstance(v,np.ndarray):
                X_tensors[k] = torch.from_numpy(v)

        for k, v in y.items():
            if k in self.encoders:
                v = self.encoders[k].transform(v)
            if isinstance(v,list):
                y_tensors[k] = torch.from_numpy(np.array(v))
            elif isinstance(v,np.ndarray):
                y_tensors[k] = torch.from_numpy(v)

        if not batchmode:
            for k, v in X_tensors.items():
                X_tensors[k] = v[0]
            for k, v in y_tensors.items():
                y_tensors[k] = v[0]
                
        return X_tensors,y_tensors

    def to_device(self,X,y,device):
        for k, v in X.items():
            X[k] = v.to(device)
        for k, v in y.items():
            y[k] = v.to(device)
        return X,y

    def from_tensor(self, X_tensors, y_tensors, batchmode=False):
        X = {}
        y = {}
        if not batchmode:
            for k, v in X_tensors.items():
                X_tensors[k] = v.reshape(1,-1)
            for k, v in y_tensors.items():
                y_tensors[k] = v.reshape(1,-1)

        for k, v in X_tensors.items():
            X[k] = v.detach().cpu().numpy()

        for k, v in y_tensors.items():
            v_np = v.detach().cpu().numpy()
            if k in self.encoders:
                v_np = self.encoders[k].inverse_transform(v_np)  # Decode encoded values
            y[k] = v_np
        
        if not batchmode:
            for k, v in X.items():
                X[k] = v[0]
            for k, v in y.items():
                y[k] = v[0]
        return X,y

    def pre_process(self,X,y):
        return X,y 

    def post_process(self,X,y):
        return X,y


class Bmode_AFC(BaseProcessing):

    def __init__(self,datasource,transform,cfg=None):
        super().__init__(datasource,transform,cfg)


    def create_split(self,split):
        if self.cfg.all_frames:
            expanded_rows = []
            for idx, row in split.iterrows():
                for frame_idx in range(row["frames"]):
                    new_row = row.copy()
                    new_row["keyframe_idx"] = frame_idx
                    expanded_rows.append(new_row)
            split = pd.DataFrame(expanded_rows)
        else:
            split = split
        return split

    def pre_process(self,X,y):
        y                           = self.split.loc[y['idx']].to_dict(orient='list')
        dataids                     = y['dataid']
        for dataid in dataids:
            data                    = self.datasource.get_data(dataid,aidx=y['keyframe_idx'])
            X.setdefault('binputs',[]).append(data['banchor'])
            if self.cfg.landmarks:
                gt_bcoords          = data['bcoords']
                indices             = np.where(data['valid'] == 1)[0]
                farthest_pair       = indices[[0,-1]]
                gt_brefcoords       = gt_bcoords[farthest_pair]    
                X.setdefault('gt_valid',[]).append(data['valid'])
                X.setdefault('gt_bcoords',[]).append(gt_bcoords)
                X.setdefault('gt_brefcoords',[]).append(gt_brefcoords)

        X['binputs']                = np.stack(X['binputs'])
        X['gt_bcoords']             = np.stack(X['gt_bcoords'])
        X['gt_brefcoords']          = np.stack(X['gt_brefcoords'])
        X['gt_valid']               = np.stack(X['gt_valid'])
        X['inputs']                 = X['binputs']

        b,h,w,c                     = X['binputs'].shape 
        if c == 1:
            X['binputs']            = X['binputs'].reshape(b,h,w)
        
        if self.cfg.landmarks:
            X['gt_coords']          = X['gt_bcoords']
            X['inputs'], (X['gt_coords'],) = self.transform(X['inputs'],keypoints_list=[X['gt_coords']],batchmode=True)
            if 'heatmap' in self.cfg and self.cfg.heatmap.type == 'gaussian':
                X['gt_normheatmaps'] = self.echotools.get_gaussian_heatmaps(
                                                    coords=X['gt_coords'],
                                                    mask =X['gt_valid'],
                                                    shape= (h,w),
                                                    heatmap_ratio=self.cfg.heatmap.ratio,
                                                    heatmap_sigma=self.cfg.heatmap.sigma,
                                                    paired=self.datasource.paired_coords,
                                                    batchmode=True,
                                                )
        else:
            X['inputs']            = self.transform(X['inputs'],keypoints_list=[],batchmode=True)
        y['bpix2cmratio']          = np.stack([y['pix2cm_y'],y['pix2cm_x']],axis=-1).reshape(-1,1,2)
        return X,y

    def post_process(self,X,y):
        X,y                        = super().post_process(X,y)
        X['pred_coords']           = self.echotools.sargmax(X['pred_normheatmaps'])
        X['pred_spreads']          = self.echotools.spreads(X['pred_normheatmaps'],X['pred_coords'])
        X['pred_bcoords']          = X['pred_coords']
        X['pred_bspreads']         = X['pred_spreads']

        lv_perp_coeffs             = []
        lv_axis_coeffs             = []
        for pred_coord in X['pred_bcoords']:
            model                  = LinearRegression()
            model.fit(pred_coord[:,1].reshape(-1,1), pred_coord[:,0])
            slope_perp             = model.coef_[0]
            intercept_perp         = model.intercept_
            pred_center            = np.mean(pred_coord,axis=0).reshape(1,2)

            if slope_perp == 0:
                slope = float('-inf')
            else:
                slope = -1 / slope_perp
            
            intercept              = pred_center[0,0] - slope * pred_center[0,1]
            slope_deg              = np.degrees(np.arctan(slope)) 
            slope_perp_deg         = np.degrees(np.arctan(slope_perp))
            lv_axis_coeffs.append([slope_deg,intercept])
            lv_perp_coeffs.append([slope_perp_deg,intercept_perp])
        lv_axis_coeffs             = np.array(lv_axis_coeffs)
        lv_perp_coeffs             = np.array(lv_perp_coeffs)

        vb                         = self.echotools.VBScanLine( 
                                        reference = lv_axis_coeffs,
                                        input_fmt='coeff',
                                        S=0,
                                        R=0,
                                        image_dim=(y['height'],y['width']),
                                        batchmode= True
                                    )
        X['pred_bSLA']             = vb.bSL(ds=10)
    
        vb                         = self.echotools.VBScanLine( 
                                        reference = lv_perp_coeffs,
                                        input_fmt='coeff',
                                        S=0,
                                        R=0,
                                        image_dim=(y['height'],y['width']),
                                        batchmode= True
                                    )
        SLcoords                   = vb.bSL()
        X['pred_bSL']              = vb.bSL(ds=10)
        vb                         = self.echotools.VBScanLine(
                                        reference = SLcoords[:,[0,-1]],
                                        S=0,
                                        R=0,
                                        image_dim=(y['height'],y['width']),
                                        batchmode= True
                                    )
                            
        X['pred_bregSL_projbcoords']= vb.B2SL_coords(X['pred_bcoords'])

        if self.cfg.landmarks:
            vb                      = self.echotools.VBScanLine(
                                        reference = X['gt_brefcoords'],
                                        S=0,
                                        R=0,
                                        image_dim=(y['height'],y['width']),
                                        batchmode= True
                                    )
            X['pred_bgtrefSL_projbcoords'] = vb.B2SL_coords(X['pred_bcoords'])
            X['gt_brefSL']          = vb.bSL(ds=10)
        else:
            pass
        return X,y



class AMM_AFC(BaseProcessing):

    def __init__(self,datasource,transform,cfg=None):
        super().__init__(datasource,transform,cfg)

    def create_split(self,split):
        return split

    def pre_process(self,X,y):
        y                      = self.split.loc[y['idx']].to_dict(orient='list')
        dataids                = y['dataid']
        bmovies                = []
        for dataid in dataids:
            data           = self.datasource.get_data(dataid)
            movie          = data['bmovie']
            bmovies.append(movie)
            X.setdefault('gt_valid',[]).append(data['valid'])
            if 'bSL' in data:
                gt_bcoords     = data['bSL']
                gt_brefcoords  = gt_bcoords[[0,-1]]
            else:
                gt_bcoords     = data['bcoords']
                indices        = np.where(data['valid'] == 1)[0]
                farthest_pair  = indices[[0,-1]]
                gt_brefcoords  = gt_bcoords[farthest_pair]
            X.setdefault('gt_bcoords',[]).append(gt_bcoords)
            X.setdefault('gt_brefcoods',[]).append(gt_brefcoords)
            X.setdefault('binputs',[]).append(data['banchor'])

        X['binputs']                = np.stack(X['binputs'])
        X['gt_bcoords']             = np.stack(X['gt_bcoords'])
        X['gt_brefcoods']           = np.stack(X['gt_brefcoods'])
        X['gt_valid']               = np.stack(X['gt_valid'])
        b,h,w,c                     = X['binputs'].shape 
        if c == 1:
            X['binputs']            = X['binputs'].reshape(b,h,w)


        bclip,aidx                  = self.echotools.clip_movie(
                                        movies=bmovies,
                                        Aidx=y['keyframe_idx'],
                                        WS=self.cfg.window_size,
                                        batchmode=True
                                    )

        vb                          = self.echotools.VBScanLine(
                                        reference = X['gt_brefcoods'],
                                        Aidx=aidx, 
                                        S=0,
                                        R=0,
                                        image_dim=(y['height'],y['width']),
                                        batchmode=True
                                    )

        X['gt_coords']              = vb.B2AMM_coords(X['gt_bcoords'])
        X['gt_mcoords']             = X['gt_coords']
        gt_brefSL                   = vb.bSL()
        
        X['gt_brefSL']              = vb.bSL(ds=10)
        X['gt_mrefSL']              = vb.mSL(ds=10)

        X['inputs']                 = self.echotools.clip2amm(
                                                            bclip,
                                                            gt_brefSL,
                                                            batchmode=True
                                                        )
        _,h,w,c                     = X['inputs'].shape
        if c == 1:
            X['minputs']            = X['inputs'].reshape(b,h,w)

        X['gt_normheatmaps']        = self.echotools.get_gaussian_heatmaps(
                                        coords=X['gt_coords'],
                                        mask = X['gt_valid'],
                                        shape= (h,w),
                                        heatmap_ratio=self.cfg.heatmap.ratio,
                                        heatmap_sigma=self.cfg.heatmap.sigma,
                                        paired= self.datasource.paired_coords,
                                        batchmode=True,
                                    )
        X['inputs'],(X['gt_coords'],)  = self.transform(X['inputs'],keypoints_list=[X['gt_coords']],batchmode=True)
        y['bpix2cmratio']           = np.stack([y['pix2cm_y'],y['pix2cm_x']],axis=-1).reshape(-1,1,2)
        y['aidx']                   = aidx
        return X,y


    def post_process(self,X,y):    
        X['pred_coords']    = self.echotools.sargmax(X['pred_normheatmaps'],y['aidx'])
        vb                  = self.echotools.VBScanLine(
                                    reference = X['gt_brefcoods'],
                                    Aidx=y['keyframe_idx'], 
                                    S=0,
                                    R=0,
                                    image_dim=(y['height'],y['width']),
                                    batchmode=True
                                )
        X['pred_bcoords']   = vb.AMM2B_coords(X['pred_coords'])
        X['pred_mcoords']   = X['pred_coords']
        X['pred_mspreads']  = self.echotools.spreads(X['pred_normheatmaps'],X['pred_coords'],y['aidx'])
        X['pred_bregSL_projbcoords'] = X['pred_bcoords']
        X['pred_bgtrefSL_projbcoords']=X['pred_bcoords']
        return X,y
