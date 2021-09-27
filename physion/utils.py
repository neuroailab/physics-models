import os
import logging
import re
import numpy as np
import torch
import mlflow
from physopt.utils import PhysOptObjective

class PytorchPhysOptObjective(PhysOptObjective):
    def __init__(self, *args, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # must set device first since used in get_model, called in super
        super().__init__(*args, **kwargs)
        self.init_seed()

    def load_model(self, model_file):
        assert os.path.isfile(model_file), f'Cannot find model file: {model_file}'
        self.model.load_state_dict(torch.load(model_file))
        logging.info(f'Loaded existing ckpt from {model_file}')
        return self.model

    def save_model(self, model_file):
        logging.info(f'Saved model checkpoint to: {model_file}')
        torch.save(self.model.state_dict(), model_file)

    def init_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    @staticmethod
    def process_results(results):
        output = []
        count = 0
        processed = set()
        for i, (stim_name, test_proba, label) in enumerate(zip(results['stimulus_name'], results['test_proba'], results['labels'])):
            if stim_name in processed:
                print('Duplicated item: {}'.format(stim_name))
            else:
                count += 1
                processed.add(stim_name)
                data = {
                    'Model': results['model_name'],
                    'Readout Train Data': results['readout_name'],
                    'Readout Test Data': results['readout_name'],
                    'Train Accuracy': results['train_accuracy'],
                    'Test Accuracy': results['test_accuracy'],
                    'Readout Type': results['protocol'],
                    'Predicted Prob_false': test_proba[0],
                    'Predicted Prob_true': test_proba[1],
                    'Predicted Outcome': np.argmax(test_proba),
                    'Actual Outcome': label,
                    'Stimulus Name': stim_name,
                    'Encoder Training Dataset': results['pretraining_name'], 
                    'Encoder Training Seed': results['seed'], 
                    'Dynamics Training Dataset': results['pretraining_name'], 
                    'Dynamics Training Seed': results['seed'], 
                    }
                data.update(get_model_attributes(results['model_name'])) # add extra metadata about model
                output.append(data)
        print('Model: {}, Train: {}, Test: {}, Type: {}, Len: {}'.format(
            results['model_name'], results['pretraining_name'], results['readout_name'], results['protocol'], count))
        return output

def get_model_attributes(model):
    frozen_pattern = 'p([A-Z]+)_([A-Z]+)'
    if model == 'CSWM':
        return {
            'Encoder Type': 'CSWM encoder',
            'Dynamics Type': 'CSWM dynamics',
            'Encoder Pre-training Task': 'null', 
            'Encoder Pre-training Dataset': 'null', 
            'Encoder Pre-training Seed': 'null', 
            'Encoder Training Task': 'Contrastive',
            'Dynamics Training Task': 'Contrastive',
            }
    elif bool(re.match(frozen_pattern, model)):
        match = re.search(frozen_pattern, model)
        return {
            'Encoder Type': match.group(1),
            'Dynamics Type': match.group(2),
            'Encoder Pre-training Task': 'ImageNet classification', 
            'Encoder Pre-training Dataset': 'ImageNet', 
            'Encoder Pre-training Seed': 'null', 
            'Encoder Training Task': 'null',
            'Encoder Training Dataset': 'null', # overwrite
            'Encoder Training Seed': 'null', # overwrite
            'Dynamics Training Task': 'L2 on latent',
            }
    elif model == 'OP3':
        return {
            'Encoder Type': 'OP3 encoder',
            'Dynamics Type': 'OP3 dynamics',
            'Encoder Pre-training Task': 'null', 
            'Encoder Pre-training Dataset': 'null', 
            'Encoder Pre-training Seed': 'null', 
            'Encoder Training Task': 'Image Reconstruction',
            'Dynamics Training Task': 'Image Reconstruction',
            }
    else:
        raise NotImplementedError
