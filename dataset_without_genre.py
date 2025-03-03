from torch.utils.data import Dataset
import pandas as pd
import torch

class MovieLensDataset(Dataset):
    def __init__(self, path=None):
        super().__init__()
        self.dataframe = self.load_movielens_data(path)

    def load_movielens_data(self,path):
        column_names = ['user_id', 'item_id', 'rating']
        data = pd.read_csv(path,
                           sep='::',names=column_names,usecols=range(0,3),engine='python')
        data['user_id'] -= 1
        data['item_id'] -= 1
        return data

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self,idx):
        user_id = self.dataframe.iloc[idx, 0]
        item_id = self.dataframe.iloc[idx, 1]
        rating = self.dataframe.iloc[idx, 2]
        return (torch.tensor(user_id,dtype=torch.long),
                torch.tensor(item_id,dtype=torch.long),
                torch.tensor(rating ,dtype=torch.float).unsqueeze(0))