import os
import pickle
import pandas as pd
import joblib
from collections import defaultdict

def get_model_data_name(args):
    if not args.relabel:
        if args.only_stock_rel:
            if args.stock_article_only:
                run_id = "d3a383a83143404da6d48283f829d9dc" # global_neg V3_stock_rel_stock_article
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_stock_rel_stock_article'
                print("V3_stock_rel_stock_article")
            else:
                run_id = "a25d8e8d943442fb91177fb75eb2862e" # global_neg V3_stock_rel
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_stock_rel'
                print("V3_stock_rel")
        else:
            if args.stock_article_only:
                run_id = "9f6e83eb51a34923a6883b6de1963e29" # global_neg V3_two_rel_stock_article
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_two_rel_stock_article'
                print("V3_two_rel_stock_article")
            else:
                run_id = "afe396c21df8417eb52985bc359c0925" # global_neg V3_two_rel
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_two_rel'
                print("V3_two_rel")
    else:
        if args.only_stock_rel:
            if args.stock_article_only:
                run_id = "23682dce93704d1ab66c17ac0f0d15bd" # hard_neg V3_stock_rel_stock_article
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_relabel_stock_rel_stock_article'
                print("relabel V3_stock_rel_stock_article")
            else:
                run_id = "63000f1a54c64263aa42a11b91b27496"
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_relabel_stock_rel'
                print("relabel V3_stock_rel")
        else:
            if args.stock_article_only:
                run_id = "16e5fa1d6c7a4ffd8fa6d7cf0b6ff62d" 
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_relabel_two_rel_stock_article'
                print("relabel V3_two_rel_stock_article")
            else:
                run_id = "c74dfd30ad5e4b008159a48812192224"
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_relabel_two_rel'
                print("relabel V3_two_rel")
    return model_uri, data_version

def get_article_data(args):
    if args.relabel:
        return pd.read_csv("/data/kg_data/metadata/20220622-20220705/relabel_ArticleData-221014.csv")
    else:
        return joblib.load("/data/kg_data/metadata/20220622-20220705/ArticleData-220713.joblib")
    
def get_user_stock_dict():
    with open("/data/kg_data/appid18_viewstock_20220622_20220705/user_stock_dict.pkl", "rb") as f:
        user_stock_dict = pickle.load(f)
    return defaultdict(list, user_stock_dict)

def get_stock_user_dict():
    with open("/data/kg_data/appid18_viewstock_20220622_20220705/user_stock_dict.pkl", "rb") as f:
        user_stock_dict = pickle.load(f)
    stock_user_dict = defaultdict(list)
    for user, stock_list in user_stock_dict.items():
        for stock in stock_list:
            stock_user_dict[stock].append(user)
    return stock_user_dict

def get_interaction_data(args):
    base = "./data"
    if args.relabel:
        data_dir = os.path.join(base, "CMoney_relabel")
    else:
        data_dir = os.path.join(base, "CMoney")
    if args.only_stock_rel:
        data_dir = os.path.join(data_dir, "V3_data_only_stock_rel")
    else:
        data_dir = os.path.join(data_dir, "V3_data_two_rel")
    if args.stock_article_only:
        data_dir = os.path.join(data_dir, "have_stock_article_only")
    id_ent_dict_path = os.path.join(data_dir, "KGCN_id_ent_dict.pkl")
    ent_id_dict_path = os.path.join(data_dir, "KGCN_ent_id_dict.pkl")

    with open(id_ent_dict_path, "rb") as f:
        id_ent_dict = pickle.load(f)
    with open(ent_id_dict_path, "rb") as f:
        ent_id_dict = pickle.load(f)
    return id_ent_dict, ent_id_dict