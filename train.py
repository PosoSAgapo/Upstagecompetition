from logging import getLogger
from recbole.config import Config
import logging
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR, SimpleX, SGL, LightGCN, NCL
from recbole.model.context_aware_recommender import AutoInt, DCNV2, WideDeep
from recbole.model.sequential_recommender import GRU4Rec, GRU4RecF, DIEN, CORE, NPE, SASRecF, FDSA
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_trainer
from recbole.utils.case_study import full_sort_topk
import torch
from recbole.data.interaction import Interaction
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='experiment setting')
parser.add_argument('--model', type=str, default='GRU4Rec')
args = parser.parse_args()

if args.model == 'GRU4Rec':
    config = Config(model='GRU4Rec', dataset='upstage', config_file_list=['GRU4Rec.yaml'])
elif args.model == 'GRU4RecF':
    config = Config(model='GRU4RecF', dataset='upstage', config_file_list=['GRU4RecF.yaml'])
elif args.model == 'DIEN':
    config = Config(model='DIEN', dataset='upstage', config_file_list=['DIEN.yaml'])
elif args.model == 'CORE':
    config = Config(model='CORE', dataset='upstage', config_file_list=['CORE.yaml'])
elif args.model == 'NPE':
    config = Config(model='NPE', dataset='upstage', config_file_list=['NPE.yaml'])
elif args.model == 'SASRecF':
    config = Config(model='SASRecF', dataset='upstage', config_file_list=['SASRecF.yaml'])
elif args.model == 'FDSA':
    config = Config(model='FDSA', dataset='upstage', config_file_list=['FDSA.yaml'])
# init random seed
init_seed(config['seed'], config['reproducibility'])

# logger initialization
init_logger(config)
logger = getLogger()
# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
logger.addHandler(c_handler)


# write config info into log
logger.info(config)

dataset = create_dataset(config)
logger.info(dataset)

train_data, valid_data, test_data = data_preparation(config, dataset)

# model loading and initialization
if args.model == 'GRU4Rec':
    model = GRU4Rec(config, train_data.dataset).to(config['device'])
elif args.model == 'GRU4RecF':
    model = GRU4RecF(config, train_data.dataset).to(config['device'])
elif args.model == 'DIEN':
    model = DIEN(config, train_data.dataset).to(config['device'])
elif args.model == 'CORE':
    model = CORE(config, train_data.dataset).to(config['device'])
elif args.model == 'NPE':
    model = NPE(config, train_data.dataset).to(config['device'])
elif args.model == 'SASRecF':
    model = SASRecF(config, train_data.dataset).to(config['device'])
elif args.model == 'FDSA':
    model = FDSA(config, train_data.dataset).to(config['device'])
logger.info(model)

# trainer loading and initialization
trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

# model training
best_valid_score, best_valid_result = trainer.fit(train_data,valid_data, saved=True, show_progress=config["show_progress"])
#
