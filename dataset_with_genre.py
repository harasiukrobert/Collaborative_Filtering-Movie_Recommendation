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


        data_movie_genre = pd.read_csv('movielensdataset/movies.dat',
                           sep='::',names=['item_id','__','genres'],usecols=[0,2],engine='python', encoding='latin1')
        data_movie_genre['item_id'] -= 1
        data_movie_genre = data_movie_genre.set_index('item_id')

        genre_dummies = data_movie_genre['genres'].str.get_dummies(sep='|')
        genre_dummies = genre_dummies.drop(columns=['(no genres listed)','IMAX'], errors='ignore')

        genre_dict_values = genre_dummies.values.tolist()
        genre_dict = dict(zip(data_movie_genre.index, genre_dict_values))

        data['genre'] = data['item_id'].map(genre_dict)
        return data

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self,idx):
        user_id = self.dataframe.iloc[idx, 0]
        item_id = self.dataframe.iloc[idx, 1]
        rating = self.dataframe.iloc[idx, 2]
        genere = self.dataframe.iloc[idx, 3]
        return (torch.tensor(user_id,dtype=torch.long),
                torch.tensor(item_id,dtype=torch.long),
                torch.tensor(rating ,dtype=torch.float).unsqueeze(0),
                torch.tensor(genere,dtype=torch.long))