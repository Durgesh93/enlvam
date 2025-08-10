import numpy as np
import albumentations as A

class PixelAug(object):

    def __init__(self,cfg):
        no_op_t            = A.NoOp()
        pixel_t            = A.OneOf([
                                A.RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=(0.2, 0.4), p=1),
                                A.RandomGamma(gamma_limit=(80, 160), p=1)
                              ], p=1)
        
        self.transform     = A.Compose(
                                [A.OneOf([no_op_t, pixel_t], p=1), A.ToFloat(max_value=255.)],
                                keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
                            )
        
    def __call__(self, image, keypoints_list=[],batchmode=False):
        if not batchmode:
            image             = np.array([image])
            if len(keypoints_list):
                for t in range(len(keypoints_list)):
                    keypoints_list[t]    =  np.array([keypoints_list[t]])
            else:
                keypoints_list           = [np.zeros((1,2,2))]

        if batchmode and len(keypoints_list)==0:
            keypoints_list               = [np.zeros((len(image),2,2))]
            empty_keypoints              = True 
        else:
            empty_keypoints              = False


        image_batch             = image.astype(np.uint8)
        keypoints_batch_list    = [k.astype(np.float32) for  k in keypoints_list]

        timage_batch            = []
        t_keypoints_batch       = np.empty((len(image_batch),len(keypoints_batch_list)),dtype=object)

        for idx in range(len(image_batch)):
            image             = image_batch[idx]
            kplist            = [k[idx].tolist() for k in keypoints_batch_list ]
            data              = {}
            data['image']     = image
            for k,kp in enumerate(kplist):
                data[f'keypoints_{k}'] = kp
            t_dict            = self.transform(**data)
            t_image           = np.moveaxis(t_dict['image'],source=(2,0,1),destination=(0,1,2))
            timage_batch.append(t_image)
            for t in range(t_keypoints_batch.shape[1]):
                t_keypoints_batch[idx,t] = np.array(t_dict[f'keypoints_{t}'])
            
        timage_batch                   = np.stack(timage_batch)
        tkeypoints_batch_list         = []
        for t in range(t_keypoints_batch.shape[1]):
            tkeypoints_batch_list.append(np.stack(t_keypoints_batch[:,t]))

        if not batchmode:
            timage_batch          = timage_batch[0]
            for t in range(len(tkeypoints_batch_list)):
                tkeypoints_batch_list[t] = tkeypoints_batch_list[t][0]
    
        if not empty_keypoints:
            return timage_batch,tkeypoints_batch_list
        else:
            return timage_batch


class Identity(PixelAug):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.transform     =  A.Compose(
                                    [A.ToFloat(max_value=255)],
                                    keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
                            )