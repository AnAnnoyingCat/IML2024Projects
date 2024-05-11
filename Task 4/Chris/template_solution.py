# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Depending on your approach, you might need to adapt the structure of this template or parts not marked by TODOs.
# It is not necessary to completely follow this template. Feel free to add more code and delete any parts that 
# are not required 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = None  # TODO: Set the batch size according to both training performance and available memory
NUM_EPOCHS = None  # TODO: Set the number of epochs

train_val = pd.read_csv("train.csv")
test_val = pd.read_csv("test_no_score.csv")

# TODO: Fill out the ReviewDataset
class ReviewDataset(Dataset):
    def __init__(self, data_frame):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


train_dataset = ReviewDataset(train_val)
test_dataset = ReviewDataset(test_val)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=16, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=16, pin_memory=True)

# Additional code if needed

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
