import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random


class DataLoader:
    '''
    Data Loader class which makes dataset for training / knowledge graph dictionary
    '''
    def __init__(self, data):
        self.cfg = {
            'movie': {
                'item2id_path': 'data/movie/item_index2entity_id.txt',
                'kg_path': 'data/movie/kg.txt',
                'rating_path': 'data/movie/ratings.csv',
                'rating_sep': ',',
                'threshold': 4.0
            },
            'music': {
                'item2id_path': 'data/music/item_index2entity_id.txt',
                'kg_path': 'data/music/kg.txt',
                'rating_path': 'data/music/user_artists.dat',
                'rating_sep': '\t',
                'threshold': 0.0
            },
            'CMoney_stock_rel_stock_article': {
                'kg_path': 'data/CMoney/V3_data_only_stock_rel/have_stock_article_only/kg_final.txt',
                'rating_path': 'data/CMoney/V3_data_only_stock_rel/have_stock_article_only/ratings_final.txt',
                'rating_sep': '\t',
                'threshold': 0.5
            }, 
            'CMoney_stock_rel': {
                'kg_path': 'data/CMoney/V3_data_only_stock_rel/kg_final.txt',
                'rating_path': 'data/CMoney/V3_data_only_stock_rel/ratings_final.txt',
                'rating_sep': '\t',
                'threshold': 0.5
            }, 
            'CMoney_two_rel': {
                'kg_path': 'data/CMoney/V3_data_two_rel/kg_final.txt',
                'rating_path': 'data/CMoney/V3_data_two_rel/ratings_final.txt',
                'rating_sep': '\t',
                'threshold': 0.5
            }, 
            'CMoney_two_rel_stock_article': {
                'kg_path': 'data/CMoney/V3_data_two_rel/have_stock_article_only/kg_final.txt',
                'rating_path': 'data/CMoney/V3_data_two_rel/have_stock_article_only/ratings_final.txt',
                'rating_sep': '\t',
                'threshold': 0.5
            }, 
            'CMoney_relabel_stock_rel_stock_article': {
                'kg_path': 'data/CMoney_relabel/V3_data_only_stock_rel/have_stock_article_only/kg_final.txt',
                'rating_path': 'data/CMoney_relabel/V3_data_only_stock_rel/have_stock_article_only/ratings_final.txt',
                'rating_sep': '\t',
                'threshold': 0.5
            }, 
            'CMoney_relabel_stock_rel': {
                'kg_path': 'data/CMoney_relabel/V3_data_only_stock_rel/kg_final.txt',
                'rating_path': 'data/CMoney_relabel/V3_data_only_stock_rel/ratings_final.txt',
                'rating_sep': '\t',
                'threshold': 0.5
            }, 
            'CMoney_relabel_two_rel': {
                'kg_path': 'data/CMoney_relabel/V3_data_two_rel/kg_final.txt',
                'rating_path': 'data/CMoney_relabel/V3_data_two_rel/ratings_final.txt',
                'rating_sep': '\t',
                'threshold': 0.5
            }, 
            'CMoney_relabel_two_rel_stock_article': {
                'kg_path': 'data/CMoney_relabel/V3_data_two_rel/have_stock_article_only/kg_final.txt',
                'rating_path': 'data/CMoney_relabel/V3_data_two_rel/have_stock_article_only/ratings_final.txt',
                'rating_sep': '\t',
                'threshold': 0.5
            }
        }
        self.data = data
        
        # df_item2id = pd.read_csv(self.cfg[data]['item2id_path'], sep='\t', header=None, names=['item','id']) #
        df_kg = pd.read_csv(self.cfg[data]['kg_path'], sep='\t', header=None, names=['head','relation','tail'])
        df_rating = pd.read_csv(self.cfg[data]['rating_path'], sep=self.cfg[data]['rating_sep'], names=['userID', 'itemID', 'rating'], skiprows=0)
        
        # df_rating['itemID'] and df_item2id['item'] both represents old entity ID
        # df_rating = df_rating[df_rating['itemID'].isin(df_item2id['item'])] #
        df_rating.reset_index(inplace=True, drop=True)
        
        # self.df_item2id = df_item2id #
        self.df_kg = df_kg
        self.df_rating = df_rating
        
        self.user_encoder = LabelEncoder()
        self.entity_encoder = LabelEncoder()
        self.relation_encoder = LabelEncoder()

        if self.data == 'Jeff_only_stock':
            self.user_encoder.fit(range(19544))
            self.entity_encoder.fit(range(37299))
            self.relation_encoder.fit([0])
        elif self.data == 'Jeff_three_rel':
            self.user_encoder.fit(range(19544))
            self.entity_encoder.fit(range(47963))
            self.relation_encoder.fit([0,1,2])
        else:
            self._encoding()
        
    def _encoding(self):
        '''
        Fit each label encoder and encode knowledge graph
        '''
        self.user_encoder.fit(self.df_rating['userID'])
        # df_item2id['id'] and df_kg[['head', 'tail']] represents new entity ID
        # self.entity_encoder.fit(pd.concat([self.df_item2id['id'], self.df_kg['head'], self.df_kg['tail']])) #
        self.entity_encoder.fit(pd.concat([self.df_rating['itemID'], self.df_kg['head'], self.df_kg['tail']]))
        self.relation_encoder.fit(self.df_kg['relation'])
        
        # encode df_kg
        self.df_kg['head'] = self.entity_encoder.transform(self.df_kg['head'])
        self.df_kg['tail'] = self.entity_encoder.transform(self.df_kg['tail'])
        self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['relation'])

    def _build_dataset(self, global_neg):
        '''
        Build dataset for training (rating data)
        It contains negative sampling process
        '''
        print('Build dataset dataframe ...', end=' ')
        # df_rating update
        df_dataset = pd.DataFrame()
        df_dataset['userID'] = self.user_encoder.transform(self.df_rating['userID'])
        
        # update to new id
        # item2id_dict = dict(zip(self.df_item2id['item'], self.df_item2id['id']))  #
        # self.df_rating['itemID'] = self.df_rating['itemID'].apply(lambda x: item2id_dict[x])  #
        df_dataset['itemID'] = self.entity_encoder.transform(self.df_rating['itemID'])
        df_dataset['label'] = self.df_rating['rating'].apply(lambda x: 0 if x <= self.cfg[self.data]['threshold'] else 1)

        ## df_dataset 會有重複的positive samples
        
        # negative sampling
        if global_neg:
            df_dataset = df_dataset[df_dataset['label']==1]  ## 這一行使得Data中label=0全都是global negative samples (不一定有impression紀錄、而是隨機sample到的)
            # df_dataset requires columns to have new entity ID
            full_item_set = set(range(len(self.entity_encoder.classes_)))
            user_list = []
            item_list = []
            label_list = []
            for user, group in df_dataset.groupby(['userID']):
                item_set = set(group['itemID'])   # 該user的positive sample的數量
                negative_set = full_item_set - item_set
                negative_sampled = random.sample(negative_set, len(item_set))  # 有多少positive samples就有多少negative samples
                user_list.extend([user] * len(negative_sampled))
                item_list.extend(negative_sampled)
                label_list.extend([0] * len(negative_sampled))
            negative = pd.DataFrame({'userID': user_list, 'itemID': item_list, 'label': label_list})
            df_dataset = pd.concat([df_dataset, negative])
            df_dataset = df_dataset.sample(frac=1, replace=False, random_state=999)
        df_dataset.reset_index(inplace=True, drop=True)
        print('Done')
        return df_dataset
        
        
    def _construct_kg(self):
        '''
        Construct knowledge graph
        Knowledge graph is dictionary form
        'head': [(relation, tail), ...]
        '''
        print('Construct knowledge graph ...', end=' ')
        kg = dict()
        for i in range(len(self.df_kg)):
            head = self.df_kg.iloc[i]['head']
            relation = self.df_kg.iloc[i]['relation']
            tail = self.df_kg.iloc[i]['tail']
            if head in kg:
                kg[head].append((relation, tail))
            else:
                kg[head] = [(relation, tail)]
            if tail in kg:
                kg[tail].append((relation, head))
            else:
                kg[tail] = [(relation, head)]
        print('Done')
        return kg
        
    def load_dataset(self, global_neg = True):
        return self._build_dataset(global_neg = global_neg)

    def load_kg(self):
        return self._construct_kg()
    
    def get_encoders(self):
        return (self.user_encoder, self.entity_encoder, self.relation_encoder)
    
    def get_num(self):
        return (len(self.user_encoder.classes_), len(self.entity_encoder.classes_), len(self.relation_encoder.classes_))
