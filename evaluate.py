import argparse
from utils import *
from data_loader import DataLoader
import numpy as np
import mlflow
import torch
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['rating'], dtype=np.float32)
        return user_id, item_id, label
    
def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--relabel', type=bool, default=True, help='use the relabeled stock tag data')
    parser.add_argument('--only_stock_rel', type=bool, default=False, help='only use relation "stock tag" in the KG')
    parser.add_argument('--stock_article_only', type=bool, default=True, help='only use article with stock tag')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--eval_frac', type=float, default=0.001, help='fraction of evaluation data')
    args = parser.parse_args()
    return args

class Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self, model_uri):
        self.net = mlflow.pytorch.load_model(model_uri).to(self.device)
        
    def load_data(self, data_version):
        data_loader = DataLoader(data_version)
        self.rating_df = data_loader.df_rating
        self.user_encoder, self.ent_encoder, _ = data_loader.get_encoders()
        self.rating_df['itemID'] = self.ent_encoder.transform(self.rating_df['itemID'])
        self.rating_df['userID'] = self.user_encoder.transform(self.rating_df['userID'])
        
    def evaluate(self, frac=0.001):
        rating_sample_df = self.rating_df.sample(frac=frac, random_state=50)
        rating_sample_dataset = KGCNDataset(rating_sample_df)
        rating_sample_loader = torch.utils.data.DataLoader(rating_sample_dataset, batch_size=args.batch_size)
        self.net.eval()
        sample_prediction = []
        for user_ids, item_ids, labels in tqdm(rating_sample_loader):
            user_ids, item_ids, labels = user_ids.to(self.device), item_ids.to(self.device), labels.to(self.device)
            outputs = self.net(user_ids, item_ids)
            sample_prediction += outputs.tolist()
        sample_label = rating_sample_df['rating'].tolist()
        auc = roc_auc_score(sample_label, sample_prediction)
        return auc

        
if __name__ == '__main__':
    args = get_parser_args()
    
    remote_server_uri = "http://192.168.121.142:5000"
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("KGCN-PyTorch")
        
    model_uri, data_version = loader_helper.get_model_data_name(args)
    
    evaluator = Evaluator(args)
    evaluator.load_data(data_version)
    evaluator.load_model(model_uri)
    auc = evaluator.evaluate(frac=args.eval_frac)
    print("auc = ", auc)
