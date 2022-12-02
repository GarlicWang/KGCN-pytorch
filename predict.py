import argparse
import mlflow
from utils import *
import torch
from data_loader import DataLoader
import numpy as np
from numba import jit
import pickle
from time import time, ctime
from collections import Counter

def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--relabel', type=bool, default=True, help='use the relabeled stock tag data')
    parser.add_argument('--only_stock_rel', type=bool, default=False, help='only use relation "stock tag" in the KG')
    parser.add_argument('--stock_article_only', type=bool, default=True, help='only use article with stock tag')
    args = parser.parse_args()
    return args

class Predictor:
    def __init__(self, args, item_num=76122):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.item_num = item_num
    
    def load_model(self, model_uri):
        print("Loading model...")
        self.net = mlflow.pytorch.load_model(model_uri).to(self.device)
        self.user_num = self.net.usr.weight.shape[0]

    def load_data(self, data_version):
        print("Loading data...")
        self.data_loader = DataLoader(data_version)
        
        self.user_stock_dict = loader_helper.get_user_stock_dict()
        
        self.id_ent_dict, self.ent_id_dict = loader_helper.get_interaction_data(self.args)
            
        self.article_user_dict = {}
        for article_ in range(self.item_num):
            self.article_user_dict[article_] = []
        for user_, article_, _ in self.data_loader.df_rating[self.data_loader.df_rating['rating']==1].values:
            self.article_user_dict[article_].append(user_)
    
    def check_meta_path(self, user, item, meta_path_thre=1):
        user_stock_list = self.user_stock_dict[user]  # stock pages seen by target user
        article_user_list = self.article_user_dict[item]  # another viewers of target article
        
        for usr_ in article_user_list:
            if usr_ not in self.user_stock_dict:
                continue
            for stock_ in user_stock_list:
                if stock_ in self.user_stock_dict[usr_] and stock_ != 'TWA00':  # TWA00 was excluded from the relabeled data
                    return True, stock_
        return False, None
        
        # meta_path_count = 0
        # for usr_ in article_user_list:
        #     if usr_ not in self.user_stock_dict:
        #         continue
        #     stock_counter = Counter()
        #     for stock_ in user_stock_list:
        #         if stock_ in self.user_stock_dict[usr_]:
        #             stock_counter[stock_] += 1
        #             meta_path_count += 1
        # return meta_path_count, stock_counter.most_common(1)
    
    def predict(self, k):  # 80 mins
        print("Start predicting...")
        print("time = ", ctime(time()))
        max_k_array = np.ones((self.user_num, k, 3)) * -1   # initialize to -1; [score, item_id, stock_id]
        user_embed = self.net.usr.weight.cpu().detach().numpy()
        item_embed = self.net.ent.weight[:76122].cpu().detach().numpy()
        print("Calculating score...")
        print("time = ", ctime(time()))
        score_array = calculate_score(user_embed, item_embed)
        print("Done")
        print("time = ", ctime(time()))
        score_tensor = torch.sigmoid(torch.Tensor(score_array))
        sorted, indices = torch.sort(score_tensor, descending=True)
        for user_id, (original_user_id, row, ids) in enumerate(zip(self.data_loader.user_encoder.classes_, sorted, indices)):
            count = 0
            if user_id % 100 == 0:
                print("user : ", user_id)
                print("time = ", ctime(time()))
            if original_user_id not in self.user_stock_dict:
                continue
            for original_item_id, score, id in zip(self.data_loader.entity_encoder.classes_, row, ids):
                if original_item_id not in self.article_user_dict:
                    continue
                check, target_stock = self.check_meta_path(original_user_id, original_item_id)  # too slow
                if check and target_stock in self.ent_id_dict:
                    target_stock_id = self.ent_id_dict[target_stock]                
                    max_k_array[user_id, count] = [score, id, target_stock_id]
                    count += 1
                if count >= k:
                    break
        return max_k_array

@jit(nopython=True)
def calculate_score(user_embed, item_embed):
    score_array = np.empty((user_embed.shape[0], item_embed.shape[0]))
    for i, user in enumerate(user_embed):
        for j, item in enumerate(item_embed):
            score_array[i,j] = (user * item).sum().item()
    return score_array 

if __name__ == "__main__":
    args = get_parser_args()
    
    remote_server_uri = "http://192.168.121.142:5000"
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("KGCN-PyTorch")
        
    model_uri, data_version = loader_helper.get_model_data_name(args)
    
    predictor = Predictor(args)
    predictor.load_model(model_uri)
    predictor.load_data(data_version)
    predict_result = predictor.predict(5)
    print(predict_result.shape)
    with open("score_array.npy", "wb") as f:
        np.save(f, predict_result)
