import os
import sys
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dotenv import find_dotenv

envpath = find_dotenv()
sys.path.insert(0,os.path.dirname(envpath))

from data.splitter import Splitter

class base:
    def __init__(self,cfg):
        self.cfg              = cfg
        self.echotools        = self.cfg.echotools.get_instance()

    def get_metadata(self):
        raise NotImplementedError('get metadata is not implemented for this object')
    
    def get_label_encoders(self):
        encoders= {column: LabelEncoder() for column in self.metadata.select_dtypes(include=['object'])}
        for k in encoders.keys():
            encoders[k].fit(list(self.metadata[k]))
        return encoders

    def __getitem__(self,dataid):
        raise  NotImplementedError('get item is not implemented for this object')

    def __len__(self):
        raise NotImplementedError('length  is not implemented for this object')


class GE_MmodePrivate_H5(base):

    def __init__(self,cfg):
        super().__init__(cfg)
        self.file       = h5py.File(self.cfg.filepath, "r")
        self.metadata   = self.get_metadata()
        self.encoders   = self.get_label_encoders()
        self.split      = Splitter(
                            self.metadata,
                            self.cfg.splitter
                        ).get_split()
        self.paired_coords = False
    
    def get_metadata(self):
        info_dict  = {}
        for k,v in self.file['info'].items():
            if v.dtype.kind == 'S':
                v  = np.array(v).astype('U')
            info_dict[k]= v
        df         = pd.DataFrame(info_dict)
        return df
    
    def get_data(self,preid_key,target):
        if target == 'valid_coords':
            return np.array(4*[1])
        else:
            return self.file[f'/data/{target}/{preid_key}'][()]

    def __len__(self):
        return len(self.metadata)
    


class Plaxlv_GEv2private_H5(base):

    def __init__(self,cfg):
        super().__init__(cfg)
        self.file       = h5py.File(self.cfg.filepath, "r")
        self.metadata   = self.get_metadata()
        self.encoders   = self.get_label_encoders()
        self.split      = Splitter(
                            self.metadata,
                            self.cfg.splitter
                        ).get_split()
        self.paired_coords = False
    
    def get_metadata(self):
        info_dict  = {}
        for k,v in self.file['info'].items():
            if v.dtype.kind == 'S':
                v  = np.array(v).astype('U')
            info_dict[k]= v
        df         = pd.DataFrame(info_dict)
        return df
    
    def get_data(self,dataid,aidx=None,A=0):
        record                = {}
        row                   = self.metadata[self.metadata['dataid'] == dataid]
        movieid               = row['movieid'].item()
        movie                 = np.array(self.file[f'data/bmovie/{movieid}'])
        aidx                  = row['keyframe_idx'].item()
        phase                 = row['phase'].item()
        bcoords               = self.file['data/bcoords'][dataid]
        record['bcoords']     = bcoords
        mask                  = np.array([1],dtype=np.uint8)
        mask                  = mask.repeat(4,axis=0)
        record['valid']       = mask
        anchor_frame          = movie[aidx]
        h,w                   = anchor_frame.shape
        record['banchor']     = anchor_frame.reshape(h,w,1)
        record['bmovie']      = movie.reshape(-1,h,w,1)
        return record

    def __len__(self):
        return len(self.metadata)