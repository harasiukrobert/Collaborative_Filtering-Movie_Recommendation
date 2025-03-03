import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset_with_genre import MovieLensDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import defaultdict
import sys


torch.set_float32_matmul_precision('high')

class Collaborative(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.MSELoss()

        self.movie_embedding = nn.Embedding(3952, 256)
        self.user_embedding = nn.Embedding(6040, 256)
        self.genere_embedding = nn.Embedding(19, 256)
        self.fc1 = nn.Linear(256 * 3, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, movie, user, genere):
        movie_embedding = self.movie_embedding(movie)
        user_embedding = self.user_embedding(user)
        genere_embedding = self.genere_embedding(genere)
        genere_embedding = torch.mean(genere_embedding, dim=1)

        combined = torch.cat([movie_embedding, user_embedding, genere_embedding], dim=1)
        rating =  self.relu(self.fc1(combined))
        rating = self.dropout(rating)
        rating = self.fc2(rating)
        return rating

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)

    def training_step(self, batch, batch_idx):
        user, movie, rating, genere = batch
        outputs = self(movie, user, genere)
        loss = self.criterion(outputs, rating)
        loss = torch.sqrt(loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user, movie, rating, genere = batch
        outputs = self(movie, user, genere)
        loss = self.criterion(outputs, rating)
        loss = torch.sqrt(loss)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for u, p, r in zip(user, outputs, rating):
            self.val_predictions.append((u.item(), p.item(), r.item()))
        return loss

    def on_validation_epoch_start(self):
        self.val_predictions = []

    def on_validation_epoch_end(self):
        threshold = 3.0
        k = 50
        user_ratings_comparison = defaultdict(list)
        user_precisions = []
        user_recalls = []
        for u, p, r in self.val_predictions:
            user_ratings_comparison[u].append((p,r))

        for user_id, user_ratings in user_ratings_comparison.items():
            all_relevant_count = 0
            for p, r in user_ratings:
                if r >= threshold:
                    all_relevant_count +=1


            sorted_ratings = sorted(user_ratings, key=lambda x: x[0],reverse=True)[:k]
            relevant_count  = 0
            for p, r in sorted_ratings:
                if p >= threshold and r>= threshold:
                    relevant_count  +=1

            precision_precent = relevant_count  / k * 100
            recall_precent = relevant_count / all_relevant_count  * 100 if all_relevant_count > 0 else 100.0

            user_precisions.append(precision_precent)
            user_recalls.append(recall_precent)

        avg_precision = sum(user_precisions) / len(user_precisions)
        avg_recall = sum(user_recalls) / len(user_recalls)

        self.log_dict({"avg_precision": avg_precision, "avg_recall": avg_recall}, logger=True)
        print(f'\navg precision: {avg_precision}%\tavg recall: {avg_recall}%')

if __name__ == '__main__':
    for i in range(1,6):
        traindataset = MovieLensDataset(f'movielensdataset/r{i}.train')
        valdatset = MovieLensDataset(f'movielensdataset/r{i}.test')

        traindataloader = DataLoader(traindataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
        valdataloader = DataLoader(valdatset, batch_size=256, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints/',
            filename='model{i}-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min',
            save_last=True
        )
        print(f'r{i}.test')

        if os.path.exists("checkpoints/"):
            files = [os.path.join('checkpoints/', file) for file in os.listdir('checkpoints/') if file.startswith('last')]
            files.sort(key=os.path.getctime,reverse=True)
            loaded_model = Collaborative.load_from_checkpoint(files[0])
        else:
            loaded_model = Collaborative()

        trainer = pl.Trainer(max_epochs=20, fast_dev_run=False, accelerator='gpu', callbacks=[checkpoint_callback])
        trainer.fit(loaded_model, traindataloader, valdataloader)
