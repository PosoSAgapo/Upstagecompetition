from logging import getLogger
from recbole.config import Config
import logging
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR, SimpleX, SGL, LightGCN, NCL
from recbole.model.context_aware_recommender import AutoInt, DCNV2
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.utils.case_study import full_sort_topk, full_sort_scores
import torch
from recbole.data.interaction import Interaction
import pandas as pd
import os
import torch
from recbole.quick_start import load_data_and_model
import argparse
from tqdm import tqdm
import numpy as np

def find_new_file(dir):
     file_lists = os.listdir(dir)
     file_lists.sort(key=lambda fn: os.path.getmtime(dir + "/" + fn)
                      if not os.path.isdir(dir + "/" + fn) else 0)
     file = os.path.join(dir, file_lists[-1])
     return file

parser = argparse.ArgumentParser(description='experiment setting')
parser.add_argument('--model', type=str, default='BPR')
args = parser.parse_args()
if args.model == 'BPR':
    model_file = find_new_file('/work/gk77/k77025/RecExample/saved')
elif args.model == 'SimpleX':
    model_file = find_new_file('/work/gk77/k77025/RecExample/SimpleX_saved')
elif args.model == 'SGL':
    model_file = find_new_file('/work/gk77/k77025/RecExample/SGL_saved')
elif args.model == 'LightGCN':
    model_file = find_new_file('/work/gk77/k77025/RecExample/LightGCN_saved')
elif args.model == 'NCL':
    model_file = find_new_file('/work/gk77/k77025/RecExample/NCL_saved')
elif args.model == 'AutoInt':
    model_file = find_new_file('/work/gk77/k77025/RecExample/AutoInt_saved')
elif args.model == 'DCNV2':
    model_file = find_new_file('/work/gk77/k77025/RecExample/DCNV2_saved')
elif args.model == 'WideDeep':
    model_file = find_new_file('/work/gk77/k77025/RecExample/WideDeep_saved')
print(model_file)
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
model_file=model_file,
)

model.eval()
if args.model in ['BPR', 'SimpleX', 'SGL', 'LightGCN','NCL']:
    uid_series = dataset.token2id(dataset.uid_field, [str(i) for i in range(5000)])
    topk_score, topk_iid_list = full_sort_topk(uid_series, model, valid_data, k=50, device=config["device"])
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
    test_df = pd.read_csv(os.path.join('/work/gk77/k77025/RecExample/nxitempred_dataset', 'test.csv'), sep='\t')
    for idx, i in enumerate(external_item_list):
        test_df['item'][(idx) * 50:(idx + 1) * 50] = [int(x) for x in i]
    test_df['item'].astype('int')
    test_df.to_csv(os.path.join('/work/gk77/k77025/RecExample/nxitempred_dataset', args.model+'_submission.csv'), sep='\t',
                   index=False, float_format='%.0f')

