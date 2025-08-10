import torch.nn as nn
import lightning as L
from hydra.utils import instantiate

class LVAM(L.LightningModule):
    
    def __init__(self,cfg):
        super().__init__()
        self.cfg                   = cfg
        self.model                 = instantiate(self.cfg.model)
        self.loss_fn               = instantiate(self.cfg.loss)
        self.metrics               = nn.ModuleDict({dl:instantiate(v) for dl,v in self.cfg.metrics.items()})
        self.save_hyperparameters(self.cfg.hparams)
       
        
    def on_test_start(self):          
        for dl,metric in self.metrics.items():
            self.metrics[dl]       = metric.to(self.device)
        
    def on_predict_start(self):
        for dl,metric in self.metrics.items():
            self.metrics[dl]       = metric.to(self.device)

    def on_fit_start(self):
        dl_list                    = self.trainer.datamodule.src
        for k,dl in dl_list.items():
            self.logger.log_table(f'split/{k}',dataframe=dl.split)
        
    def on_validation_epoch_start(self):
        dl_list                    = self.trainer.datamodule.filter('val')
        for dl in dl_list:
            self.metrics[dl.id].reset()
    
    def on_test_epoch_start(self):
        self.outputs               = {}
        dl_list                    = self.trainer.datamodule.filter('test')
        for dl in dl_list :
            self.metrics[dl.id].reset()


    def on_validation_epoch_end(self):
        dl_list              = self.trainer.datamodule.filter('val')
        for dl in dl_list :
            processing       = self.trainer.datamodule.src[dl.id]
            X,y              = self.metrics[dl.id].compute()
            X,y              = processing.from_tensor(X,y,batchmode=True)

            if not dl.id.startswith('val/plotting/'):
                scores           =  self.metrics[dl.id].compute_eval(
                                        X,
                                        y,
                                        paired=processing.datasource.paired_coords,
                                        batch_reduction=True
                                    )

                log_metrics  = {}
                predicted_scores = scores['predicted']
                for k,score in predicted_scores.items():
                    idx = dl.id.replace('val/','')
                    log_metrics[f'{idx}'+'/predicted/'+k] = score

                projected_scores = scores['projected']
                for k,score in projected_scores.items():
                    idx = dl.id.replace('val/','')
                    log_metrics[f'{idx}'+'/projected/'+k] = score
                self.log_dict(log_metrics,on_epoch=True,on_step=False,sync_dist=True)
            else:
                if dl.id == 'val/plotting/bmode' and self.global_rank == 0:
                    idx = dl.id.replace('val/plotting','')
                    image = self.metrics[dl.id].plot_eval_bmode(X,y,processing)
                    self.logger.log_image('prediction_banchor_frame',images=[image]) 
                elif dl.id == 'val/plotting/ammmode' and self.global_rank == 0:
                    idx = dl.id.replace('val/plotting','')
                    image = self.metrics[dl.id].plot_eval_ammmode(X,y,processing)
                    self.logger.log_image('prediction_amm_image',images=[image])
            

    def on_test_epoch_end(self):
        dl_list                       = self.trainer.datamodule.filter('test')
        for dl in dl_list :
            processing                = self.trainer.datamodule.src[dl.id]
            X,y                       = self.metrics[dl.id].compute()
            X,y                       = processing.from_tensor(X,y,batchmode=True)
    
      
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dl                        = self.trainer.datamodule.filter('val')[dataloader_idx]
        processing                = self.trainer.datamodule.src[dl.id]
        X, y                      = batch['X'], batch['y']
        X,y                       = processing.from_tensor(X,y,batchmode=True)
        X,y                       = processing.pre_process(X,y)
        X,y                       = processing.to_tensor(X,y,batchmode=True)
        X,y                       = processing.to_device(X,y,self.device)
        X                         = self.model(X)
        X,y                       = processing.from_tensor(X,y,batchmode=True)
        X,y                       = processing.post_process(X,y)
        X,y                       = processing.to_tensor(X,y,batchmode=True)
        self.metrics[dl.id].update(X,y)  


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        dl                        = self.trainer.datamodule.filter('test')[dataloader_idx]
        processing                = self.trainer.datamodule.src[dl.id]
        X, y                      = batch['X'], batch['y']
        X,y                       = processing.from_tensor(X,y,batchmode=True)
        X,y                       = processing.pre_process(X,y)
        X,y                       = processing.to_tensor(X,y,batchmode=True)
        X,y                       = processing.to_device(X,y,self.device)
        X                         = self.model(X)
        X,y                       = processing.from_tensor(X,y,batchmode=True)
        X,y                       = processing.post_process(X,y)
        X,y                       = processing.to_tensor(X,y,batchmode=True)
        self.metrics[dl.id].update(X,y)

              
    def training_step(self, batch, batch_idx):
        dl                        = self.trainer.datamodule.filter('train')[0]
        batch                     = batch[0]
        X_l,y_l                   = batch['X'],batch['y']
        processing                = self.trainer.datamodule.src[dl.id]
        X_l,y_l                   = processing.from_tensor(X_l,y_l,batchmode=True)
        X_l, y_l                  = processing.pre_process(X_l,y_l)
        X_l, y_l                  = processing.to_tensor(X_l,y_l,batchmode=True)
        X_l, y_l                  = processing.to_device(X_l,y_l,self.device)
        X_l                       = self.model(X_l)
        loss,batch_out            = self.loss_fn(X_l)
        results                   = {}
        for k, v in batch_out.items():
            results[f'train/{k}'] = v.item()
        self.log_dict(results,on_step=False,on_epoch=True, prog_bar=False, logger=True,sync_dist=True)
        return loss

    def configure_optimizers(self):
        self.opt1 = instantiate(self.cfg.optimizer,params=self.parameters())
        self.sc1  = instantiate(self.cfg.scheduler,optimizer=self.opt1)
        return {'optimizer': self.opt1,'lr_scheduler': self.sc1}