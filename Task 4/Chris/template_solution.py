# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AlbertModel
import pickle
from sklearn.decomposition import PCA
import torch.nn.functional as F

# Depending on your approach, you might need to adapt the structure of this template or parts not marked by TODOs.
# It is not necessary to completely follow this template. Feel free to add more code and delete any parts that 
# are not required 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 10
tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")

train_val = pd.read_csv("Task 4/Data/train.csv")
test_val = pd.read_csv("Task 4/Data/test_no_score.csv")

class TrainReviewDataset(Dataset):
    def __init__(self, data_frame, tokenizer):
        self.data = data_frame.values 
        self.length = len(self.data)
        self.tokenizer = tokenizer

    def __len__(self):
        return self.length  

    def __getitem__(self, index):
        title, sentence, score = self.data[index]
        text = title + ": " + sentence

        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=512
        )

        return {'input_ids': encoded_dict['input_ids'], 'score': score}


class TestReviewDataset(Dataset):
    def __init__(self, data_frame, tokenizer):
        self.data = data_frame.values 
        self.length = len(self.data)
        self.tokenizer = tokenizer

    def __len__(self):
        return self.length  

    def __getitem__(self, index):
        title, sentence = self.data[index]
        text = title + ": " + sentence

        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=512
        )

        return {'input_ids': encoded_dict['input_ids']}

class EmbeddedDataset(Dataset):
    def __init__(self, embeddings, scores):
        self.embeddings = embeddings
        self.scores = scores

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx]).float()
        score = torch.tensor(self.scores[idx]).float()
        return embedding, score

train_dataset = TrainReviewDataset(train_val, tokenizer)
test_dataset = TestReviewDataset(test_val, tokenizer)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4, pin_memory=True)


if __name__ == '__main__':
    
    #the embeddings get generated in the train/testloader directly.
    print("starting the embedding process")
    model = AlbertModel.from_pretrained("albert/albert-base-v2")
    model = model.to(DEVICE) #move it to gpu
    model.eval()
    embeddings_list = []
    scores_list = []

    #init pca
    pca = PCA(n_components=64)
    
    for batch in tqdm(test_loader, desc="Processing batches"):
        # move the stuff to GPU
        input_ids = batch['input_ids'].to(DEVICE)
        currentSize = input_ids.size(0)
        if currentSize == 64:
            #forward pass
            input_ids = input_ids.squeeze(1)
            attention_mask = (input_ids != tokenizer.pad_token_id).float()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            #get embeddings
            embeddings = outputs.last_hidden_state.cpu().numpy()

            # apply dimensionality reduction
            reshaped_embeddings = embeddings.reshape(embeddings.shape[0], -1)
            reduced_embeddings = pca.fit_transform(reshaped_embeddings)

            #scores = batch['score'].tolist()

            embeddings_list.append(reduced_embeddings)
            #scores_list.extend(scores)
        else:
            pca = PCA(n_components=input_ids.size(0))
            #forward pass
            input_ids = input_ids.squeeze(1)
            attention_mask = (input_ids != tokenizer.pad_token_id).float()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            #get embeddings
            embeddings = outputs.last_hidden_state.cpu().numpy()

            # apply dimensionality reduction
            reshaped_embeddings = embeddings.reshape(embeddings.shape[0], -1)
            reduced_embeddings = pca.fit_transform(reshaped_embeddings)

            #scores = batch['score'].tolist()

            embeddings_list.append(reduced_embeddings)
            #scores_list.extend(scores)
    #embeddings = np.concatenate(embeddings_list, axis=0)
    #scores = np.array(scores_list)

    #np.savez("embeddingsAndScoresPCAReduced.npy", embeddings=embeddings, scores=scores)
    np.savez("embeddingsAndScoresPCAReduced.npy", embeddings=embeddings_list)

    print("Saved embeddings and scores to embeddingsAndScores.npy")

    """
    

    file_path = "Task 4\Chris\embeddingsAndScoresPCAReduced.npy.npz"

    data = np.load(file_path)

    embeddings = data['embeddings']
    scores = data['scores']
    print("Embeddings shape:", embeddings.shape)
    print("Scores shape:", scores.shape)

    embedded_dataset = EmbeddedDataset(embeddings, scores)

    train_loader = DataLoader(dataset=embedded_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # TODO: Fill out MyModule
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

    model = MyModule()

    scheduler = None
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    model.train()
    model.to(DEVICE)
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0 
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y.unsqueeze(1).type(torch.float))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {epoch_loss / len(train_loader):.4f}")
"""
           

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