# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DistilBertModel
import pickle
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torch.utils.data as data_utils

# Depending on your approach, you might need to adapt the structure of this template or parts not marked by TODOs.
# It is not necessary to completely follow this template. Feel free to add more code and delete any parts that 
# are not required 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 10
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class TrainReviewDataset(Dataset):
    def __init__(self, data_frame, tokenizer, max_length = 128):
        self.data = data_frame.values 
        self.length = len(self.data)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.length  

    def __getitem__(self, index):
        title, sentence, score = self.data[index]
        text = title + ": " + sentence

        encoded_dict = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoded_dict['input_ids'].squeeze(0),
            'attention_mask': encoded_dict['attention_mask'].squeeze(0),
            'score': torch.tensor(score, dtype=torch.float) 
        }


class TestReviewDataset(Dataset):
    def __init__(self, data_frame, tokenizer, max_length=128):
        self.data = data_frame.values 
        self.length = len(self.data)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.length  

    def __getitem__(self, index):
        title, sentence = self.data[index]
        text = title + ": " + sentence

        encoded_dict = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoded_dict['input_ids'].squeeze(0),
            'attention_mask': encoded_dict['attention_mask'].squeeze(0),
        }

class EmbeddedDataset(Dataset):
    def __init__(self, embeddings, scores):
        self.embeddings = embeddings
        self.scores = scores

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        score = self.embeddings[idx]
        return embedding, score
        
def makeEmbeddings():
    train_val = pd.read_csv("Task 4/Data/train.csv")
    test_val = pd.read_csv("Task 4/Data/test_no_score.csv")

    train_dataset = TrainReviewDataset(train_val, tokenizer)
    test_dataset = TestReviewDataset(test_val, tokenizer)

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)
    #the embeddings get generated in the train/testloader directly.
    print("starting the embedding process")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model = model.to(DEVICE) #move it to gpu
    model.eval()
    embeddings_list_mean = []
    embeddings_list_max = []
    embeddings_list_cls = []
    scores_list = []
    
    for batch in tqdm(train_loader, desc="Processing batches"):
        # move the stuff to GPU
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)

        #forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        #get embeddings
        last_hidden_states = torch.tensor(outputs.last_hidden_state.cpu())

        #get mean, max and cls
        #mean_embedding = torch.mean(last_hidden_states, dim=1)
        #max_embedding, _ = torch.max(last_hidden_states, dim=1)
        cls_embedding = last_hidden_states[:, 0, :]

        scores = batch['score'].tolist()

        #embeddings_list_mean.append(mean_embedding)
        #embeddings_list_max.append(max_embedding)
        embeddings_list_cls.append(cls_embedding)
        scores_list.extend(scores)

    last_tensor = embeddings_list_cls[-1]

    #pad last tensor
    padding_size = (64 - last_tensor.size(0), last_tensor.size(1))
    padded_last_tensor = torch.cat([last_tensor, torch.zeros(*padding_size)], dim=0)
    embeddings_list_cls[-1] = padded_last_tensor
    concatenated_embeddings_cls = torch.cat(embeddings_list_cls, dim=0)
    #remove padding
    concatenated_embeddings_cls = concatenated_embeddings_cls[:(-1*padding_size[0])]

    scores = np.array(scores_list)

    #np.savez("distilibertEmbeddings_mean.npy", embeddings=concatenated_embeddings_mean, scores=scores)
    #np.savez("distilibertEmbeddings_max.npy", embeddings=concatenated_embeddings_max, scores=scores)
    np.savez("distilibertEmbeddings_cls.npy", embeddings=concatenated_embeddings_cls, scores=scores)

    print("Saved embeddings and scores")


if __name__ == '__main__':
    file_path = "Task 4\Chris\distilbertEmbeddings_mean.npy.npz"
    data = np.load(file_path)
    embeddings = data['embeddings']
    scores = data['scores']

    embedded_dataset = EmbeddedDataset(embeddings, scores)

    val_size = int(0.2 * len(embedded_dataset))
    train_size = len(embedded_dataset) - val_size

    train_dataset, val_dataset = data_utils.random_split(embedded_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = MyModule()
    scheduler = None
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    model.train()
    model.to(DEVICE)
    for epoch in range(NUM_EPOCHS):

        #train it
        epoch_loss = 0 
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train.unsqueeze(1).type(torch.float))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)

        #get validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                val_output = model(X_val)
                val_loss += criterion(val_output, y_val.unsqueeze(1).type(torch.float)).item()
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")


"""
    model.eval()
    with torch.no_grad():
        results = []
        for batch in tqdm(test_loader, total=len(test_loader)):
            batch = batch.to(DEVICE)

            # TODO: Set up evaluation loop

        with open("result.txt", "w") as f:
            for val in np.concatenate(results):
                f.write(f"{val}\n")
"""