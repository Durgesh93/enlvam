import sys
import os
from dotenv import find_dotenv

envpath = find_dotenv()
sys.path.insert(0,os.path.dirname(envpath))

import conf
from hydra.utils import instantiate

def traineval_gilbert_bmode(overrides=[]):
    os.environ['ENVIRONMENT'] = 'lumid'
    EXP_NAME   = 'experiments/gilbert/bmode'
    cfg        = conf.build_config(config_path=EXP_NAME,cli_overrides=overrides)
    trainer    = instantiate(cfg.trainer)
    datamodule = instantiate(cfg.datamodule)
    b_model    = instantiate(cfg.litmodel)
    trainer.fit(b_model, datamodule=datamodule)


def traineval_gilbert_ammmode(overrides=[]):
    os.environ['ENVIRONMENT'] = 'lumid'
    EXP_NAME   = 'experiments/gilbert/ammmode'
    cfg        = conf.build_config(config_path=EXP_NAME,cli_overrides=overrides)
    trainer    = instantiate(cfg.trainer)
    datamodule = instantiate(cfg.datamodule)
    amm_model    = instantiate(cfg.litmodel)
    trainer.fit(amm_model, datamodule=datamodule)

if __name__ == '__main__':
    traineval_gilbert_bmode()
