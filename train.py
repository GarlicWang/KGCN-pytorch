import argparse
from utils import *
from data_loader import DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from model import KGCN
import mlflow
from tqdm import tqdm

class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label

def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--relabel', type=bool, default=True, help='use the relabeled stock tag data')
    parser.add_argument('--only_stock_rel', type=bool, default=False, help='only use relation "stock tag" in the KG')
    parser.add_argument('--stock_article_only', type=bool, default=True, help='only use article with stock tag')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    parser.add_argument('--n_epochs', type=int, default=3, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--train_data_ratio', type=float, default=0.8, help='size of training dataset')
    args = parser.parse_args()
    return args

class trainer:
    def __init__(self, args):
        self.args = args
    
    def load_data(self, data_version):
        self.data_loader = DataLoader(data_version)
        self.kg = self.data_loader.load_kg()
        self.df_dataset = self.data_loader.load_dataset(global_neg = True)
        
    def train(self, run_name, description, tags):
        train_loss_list = []
        test_loss_list = []
        train_auc_list = []
        test_auc_list = []
        
        # train test split
        x_train, x_test = train_test_split(self.df_dataset, test_size=1 - self.args.train_data_ratio, shuffle=False)
        train_dataset = KGCNDataset(x_train)
        test_dataset = KGCNDataset(x_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size)
        
        # prepare network, loss function, optimizer
        num_user, num_entity, num_relation = self.data_loader.get_num()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = KGCN(num_user, num_entity, num_relation, self.kg, self.args, device).to(device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.l2_weight)

        with mlflow.start_run(run_name=run_name, description=description):
            ### log params
            params = vars(self.args)
            mlflow.log_params(params)

            ### log tags
            mlflow.set_tags(tags)

            for epoch in range(self.args.n_epochs):
                train_running_loss = 0.0
                train_total_roc = 0
                for user_ids, item_ids, labels in tqdm(train_loader):
                    user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = self.net(user_ids, item_ids)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    optimizer.step()

                    train_running_loss += loss.item()
                    train_total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                
                # print train loss per every epoch
                train_loss, train_auc = train_running_loss / len(train_loader), train_total_roc / len(train_loader)
                print('[Epoch {}]train_loss: '.format(epoch+1), train_loss, ' ; train_auc: ', train_auc)
                train_loss_list.append(train_loss)
                train_auc_list.append(train_auc)
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_auc", train_auc, step=epoch)
                    
                # evaluate per every epoch
                with torch.no_grad():
                    test_running_loss = 0
                    test_total_roc = 0
                    for user_ids, item_ids, labels in tqdm(test_loader):
                        user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                        outputs = self.net(user_ids, item_ids)
                        test_running_loss += criterion(outputs, labels).item()
                        test_total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    
                    # print test loss per every epoch
                    test_loss, test_auc = test_running_loss / len(test_loader), test_total_roc / len(test_loader)
                    print('[Epoch {}]test_loss: '.format(epoch+1), test_loss, ' ; test_auc: ', test_auc)
                    test_loss_list.append(test_loss)
                    test_auc_list.append(test_auc)
                    mlflow.log_metric("test_loss", test_loss, step=epoch)
                    mlflow.log_metric("test_auc", test_auc, step=epoch)

            mlflow.pytorch.log_model(self.net, f"KGCN_epoch_{epoch+1}", registered_model_name=None)
            
if __name__ == "__main__":
    args = get_parser_args()

    model_uri, data_version = loader_helper.get_model_data_name(args)
    
    trainer = trainer(args)
    trainer.load_data(data_version)
    
    # mlflow setting
    remote_server_uri = "http://192.168.121.142:5000"
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("KGCN-PyTorch")    
    run_name = "V1 : global neg sample"
    description="random data splitting\nglobal negative sampling"
    tags = {
            "data spliting": "random", 
            "data version": "V3_relabel_stock_rel_stock_article", 
            "data date": "0622-0705", "neg sample type": "all global", 
            "neg sample num": "same as pos sample for each user (if enough)"
        }
    
    trainer.train(run_name, description, tags)