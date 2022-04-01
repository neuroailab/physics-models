import os
import csv
import pickle

import logging
import re
import numpy as np
import torch
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from torch.utils.data import DataLoader
from physopt.objective import ReadoutObjectiveBase, PhysOptModel

class PytorchModel(PhysOptModel):
    def __init__(self, *args, **kwargs): # TODO: better to not use init for this?
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # must set device first since used in get_model, called in super
        super().__init__(*args, **kwargs)

    def load_model(self, model_file):
        assert os.path.isfile(model_file), f'Cannot find model file: {model_file}'
        self.model.load_state_dict(torch.load(model_file))
        logging.info(f'Loaded existing ckpt from {model_file}')
        return self.model

    def save_model(self, model_file):
        logging.info(f'Saved model checkpoint to: {model_file}')
        torch.save(self.model.state_dict(), model_file)

    def init_seed(self):
        super().init_seed()
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    def get_dataloader(self, TDWDataset, datapaths, random_seq, shuffle, num_workers=2):
        cfg = self.pretraining_cfg
        dataset = TDWDataset(
            data_root=datapaths,
            imsize=cfg.DATA.IMSIZE,
            seq_len=cfg.DATA.SEQ_LEN,
            state_len=cfg.DATA.STATE_LEN,
            random_seq=random_seq,
            debug=self.cfg.DEBUG,
            subsample_factor=cfg.DATA.SUBSAMPLE_FACTOR,
            seed=self.seed,
            )
        dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=num_workers)
        return dataloader

class PhysionReadoutObjective(ReadoutObjectiveBase):
    def get_readout_model(self):
        steps = [('clf', LogisticRegression(max_iter=self.readout_cfg.MODEL.MAX_ITER))]
        if self.readout_cfg.MODEL.NORM_INPUT:
            steps.insert(0, ('scale', StandardScaler()))
        logging.info(f'Readout model steps: {steps}')
        pipe = Pipeline(steps)

        assert len(self.readout_cfg.MODEL.LOGSPACE) == 3, 'logspace must contain start, stop, and num'
        grid_search_params = {
            'clf__C': np.logspace(*self.readout_cfg.MODEL.LOGSPACE),
            }
        skf = StratifiedKFold(n_splits=self.readout_cfg.MODEL.CV, shuffle=True, random_state=self.seed)
        logging.info(f'CV folds: {skf}')
        readout_model = GridSearchCV(pipe, param_grid=grid_search_params, cv=skf, verbose=3)
        return readout_model

    @staticmethod
    def get_readout_model_info(readout_model):
        results = {}
        results['best_params'] = readout_model.best_params_
        results['cv_results'] = readout_model.cv_results_
        return results

    def call(self, args):
        super().call(args)
        # create physion-style CSV of model results
        for protocol in self.readout_cfg.PROTOCOLS:
            results_file = os.path.join(self.output_dir, protocol+'_metrics_results.pkl')
            results = pickle.load(open(results_file, 'rb'))
            processed_results = process_results(results, True) # change to old names
            metrics_file = os.path.join(self.output_dir, f'{results["model_name"]}-{results["readout_name"]}-results.csv')
            write_metrics(processed_results, metrics_file)
        mlflow.log_artifact(metrics_file, artifact_path='metrics')

def change_names(results): # changes to old names
    protocol_names = {
        'observed': 'A',
        'simulated': 'B',
        'input': 'C',
        }
    scenario_names = {
        'Collide': 'collision',
        'Contain': 'containment',
        'Dominoes': 'dominoes',
        'Drape': 'clothiness',
        'Drop': 'drop',
        'Link': 'linking',
        'Roll': 'rollingsliding',
        'Support': 'towers',
        }

    for key in ['readout_name', 'pretraining_name']:
        for k,v in scenario_names.items():
            if k in results[key]:
                results[key] = results[key].replace(k, v) # not very robust

    results['protocol'] = protocol_names[results['protocol']]
    return results

def process_results(results, old_names=False):
    if old_names:
        results = change_names(results)
    output = []
    count = 0
    processed = set()
    for i, (stim_name, test_proba, label) in enumerate(zip(results['stimulus_name'], results['test_proba'], results['labels'])):
        if stim_name in processed:
            logging.info('Duplicated item: {}'.format(stim_name))
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
    logging.info('Model: {}, Train: {}, Test: {}, Type: {}, Len: {}'.format(
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

def write_metrics(results, metrics_file):
    file_exists = os.path.isfile(metrics_file) # check before opening file
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames = list(results[0].keys()))
        if not file_exists: # only write header once - if file doesn't exist yet
            writer.writeheader()
        writer.writerows(results)

    logging.info('%d results written to %s' % (len(results), metrics_file))
