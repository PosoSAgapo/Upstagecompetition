from logging import getLogger
from recbole.config import Config
import logging
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR, SimpleX, SGL, LightGCN, NCL
from recbole.model.context_aware_recommender import AutoInt, DCNV2
from recbole.model.sequential_recommender import GRU4Rec, GRU4RecF, DIEN, CORE, NPE
import pickle
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

def add_last_item(old_interaction, last_item_id, max_len=50):
    new_seq_items = old_interaction['item_id_list'][-1]
    if old_interaction['item_length'][-1].item() < max_len:
        new_seq_items[old_interaction['item_length'][-1].item()] = last_item_id
    else:
        new_seq_items = torch.roll(new_seq_items, -1)
        new_seq_items[-1] = last_item_id
    return new_seq_items.view(1, len(new_seq_items))

def predict_for_all_item(external_user_id, dataset, model):
    model.eval()
    with torch.no_grad():
        uid_series = dataset.token2id(dataset.uid_field, [external_user_id])
        #index = np.isin(dataset[dataset.uid_field].numpy(), uid_series)
        index = np.isin(dataset.inter_feat['user_id'].numpy(), uid_series)
        input_interaction = dataset[index]
        try:
            last_item_id = input_interaction['item_id'][-1].item()
            item_id_list = add_last_item(input_interaction,
                                          last_item_id,
                                          model.max_seq_length)
            item_length = torch.tensor(
                [input_interaction['item_length'][-1].item() + 1
                 if input_interaction['item_length'][-1].item() < model.max_seq_length else model.max_seq_length])
        except IndexError:
            item_id_list = torch.tensor([[0]])
            item_length = torch.tensor([1])
        test = {
            'item_id_list': item_id_list,
            'item_length': item_length
        }
        new_inter = Interaction(test)
        new_inter = new_inter.to(config['device'])
        new_scores = model.full_sort_predict(new_inter)
        new_scores = new_scores.view(-1, test_data.dataset.item_num)
        new_scores[:, 0] = -np.inf  # set scores of [pad] to -inf
    return torch.topk(new_scores, 50)

parser = argparse.ArgumentParser(description='experiment setting')
parser.add_argument('--model', type=str, default='GRU4Rec')
args = parser.parse_args()
if args.model == 'GRU4Rec':
    model_file = find_new_file('/work/gk77/k77025/UpstageCompetition/saved')
elif args.model == 'GRU4RecF':
    model_file = find_new_file('/work/gk77/k77025/UpstageCompetition/GRU4RecF_saved')
elif args.model == 'GRU4RecF':
    model_file = find_new_file('/work/gk77/k77025/UpstageCompetition/DIEN_saved')
elif args.model == 'CORE':
    model_file = find_new_file('/work/gk77/k77025/UpstageCompetition/CORE_saved')
elif args.model == 'NPE':
    model_file = find_new_file('/work/gk77/k77025/UpstageCompetition/NPE_saved')
elif args.model == 'SASRecF':
    model_file = find_new_file('/work/gk77/k77025/UpstageCompetition/SASRecF_saved')
elif args.model == 'FDSA':
    model_file = find_new_file('/work/gk77/k77025/UpstageCompetition/FDSA_saved')
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
model_file=model_file,
)
f = open('test_user.pkl', 'rb')
test_user_id = pickle.load(f)
model.cuda()
f.close()
model.eval()
topk_items = []
test_df = pd.read_csv(os.path.join('/work/gk77/k77025/UpstageCompetition/submission', 'sample_submission.csv'), sep=',')
for idx, external_user_id in tqdm(enumerate(test_user_id)):
    _, topk_iid_list = predict_for_all_item(str(external_user_id), dataset, model)
    last_topk_iid_list = topk_iid_list[-1]
    external_item_list = dataset.id2token(dataset.iid_field, last_topk_iid_list.cpu()).tolist()
    topk_items.append(external_item_list)
    test_df['item_id'][(idx) * 50:(idx + 1) * 50] = [int(x) for x in external_item_list]
    test_df['rank'][(idx) * 50:(idx + 1) * 50] = list(range(50))
test_df['item_id'].astype('int')
test_df['rank'].astype('int')
test_df.to_csv(os.path.join('/work/gk77/k77025/UpstageCompetition/submission', args.model + '_submission.csv'),
    sep=',',
    index=False, float_format='%.0f')
