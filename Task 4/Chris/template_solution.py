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
            max_length=MAX_LENGTH
        )

        return {'input_ids': encoded_dict['input_ids'], 'score': score}


class TestReviewDataset(Dataset):
    def __init__(self, data_frame, tokenizer):
        self.data = data_frame.values 
        self.length = len(self.data)

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
            max_length=MAX_LENGTH
        )

        return {'input_ids': encoded_dict['input_ids']}

train_dataset = TrainReviewDataset(train_val, tokenizer)
test_dataset = TestReviewDataset(test_val, tokenizer)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=8, pin_memory=True)


#the embeddings get generated in the train/testloader directly.
print("starting the embedding process")
model = AlbertModel.from_pretrained("albert/albert-base-v2")
model = model.to(DEVICE) #move it to gpu
model.eval()
i = 0
data_to_save = []

for batch in train_loader:
    print(f("starting batch {i}"))
    i+=1
    #move the stuff to GPU
    input_ids = batch['input_ids'].to(DEVICE)

    #forward pass
    with torch.no_grad():
        outputs = model(input_ids)

    #get embeddings
    embeddings = outputs.last_hidden_state

    scores = batch['score'].tolist()
    for emb, score in zip(embeddings, scores):
        data_to_save.append({'embeddings': emb.cpu().numpy(), 'score': score})
    
output_file = "embeddingsAndScores.pkl"

with open(output_file, 'wb') as f:
    pickle.dump(data_to_save, f)

print("Saved to:", output_file)


"""

# TODO: Fill out MyModule
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


model = MyModule().to(DEVICE)

# TODO: Setup loss function, optimiser, and scheduler
criterion = None
optimiser = None
scheduler = None

model.train()
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in tqdm(train_loader, total=len(train_loader)):
        batch = batch.to(DEVICE)

        # TODO: Set up training loop


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