import os
import logging
import numpy as np
import torch
import mlflow
from physopt.utils import PhysOptObjective

class PytorchPhysOptObjective(PhysOptObjective):
    def __init__(self, *args, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # must set device first since used in get_model, called in super
        super().__init__(*args, **kwargs)
        self.init_seed()

    def load_model(self):
        if os.path.isfile(self.model_file): # load existing model ckpt TODO: add option to disable reloading
            self.model.load_state_dict(torch.load(self.model_file))
            logging.info('Loaded existing ckpt')
        else:
            torch.save(self.model.state_dict(), self.model_file) # save initial model
            logging.info('No model found, saved initial model')
        return self.model

    def save_model(self, step):
        torch.save(self.model.state_dict(), self.model_file)
        logging.info('Saved model checkpoint to: {}'.format(self.model_file))
        step_model_file = '_{}.'.format(step).join(self.model_file.split('.')) # create model file with step 
        torch.save(self.model.state_dict(), step_model_file)
        mlflow.log_artifact(step_model_file, artifact_path='model_ckpts')

    def init_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

