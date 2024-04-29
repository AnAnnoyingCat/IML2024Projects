# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")
# When using the GPU, it is important that your model and all data are on the 
# same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # Using SwinTransformer due to its recency and good performance on various tasks
    
    train_transforms = transforms.Compose([
        transforms.Resize(size=238, interpolation=transforms.InterpolationMode.BICUBIC), 
        transforms.CenterCrop(size=224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]);

    train_dataset = datasets.ImageFolder(root="Task 3/Data/dataset/", transform=train_transforms)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=False,
                              pin_memory=True, num_workers=8)

    model = torchvision.models.swin_b()

    #removing classification layer
    embedding_model = torch.nn.Sequential(*(list(model.children())[:-1]))
    #move my model to GPU if present
    embedding_model.to(device)


    embeddings = []
    i = 0
    for batch, _ in train_loader:
        inputs = batch.to(device) #move to GPU if available
        with torch.no_grad():
            #calculate batch
            batch_embeddings = embedding_model(inputs)
            embeddings.append(batch_embeddings.to(torch.device('cpu'))) #move back to CPU
            print(f"finished batch {i}")
            i+=1

    embeddings = torch.cat(embeddings, dim=0)

    embeddings_np = embeddings.numpy()

    np.save('Task 3/Chris/embeddings.npy', embeddings_np)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="Task 3/Data/dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '')[-5:] for s in train_dataset.samples]
    embeddings = np.load('Task 3/Chris/embeddings.npy')
    # TODO: Normalize the embeddings

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# Hint: adjust batch_size and num_workers to your PC configuration, so that you 
# don't run out of memory (VRAM if on GPU, RAM if on CPU)
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        # Attention: If you get type errors you can modify the type of the
        # labels here
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader

# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc = nn.Linear(3072, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = self.fc(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x

def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    n_epochs = 10
    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part 
    # of the training data as a validation split. After each epoch, compute the loss on the 
    # validation split and print it out. This enables you to see how your model is performing 
    # on the validation data before submitting the results on the server. After choosing the 
    # best model, train it on the whole training data.
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    for epoch in range(n_epochs): 
        epoch_loss = 0       
        for [X, y] in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y.unsqueeze(1).type(torch.float))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        
        print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}")
    return model

def evaluate_model(model, loader):
    """
    Evaluate the model on validation or test data.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the validation or test data
    
    output: val_loss: float, the average loss on the validation or test data
    """
    model.eval()
    criterion = nn.BCELoss()
    val_loss = 0
    with torch.no_grad():
        for [X, y] in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y.unsqueeze(1).type(torch.float))
            val_loss += loss.item()
    return val_loss

def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("Task 3/Chris/results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'Task 3/Data/train_triplets.txt'
    TEST_TRIPLETS = 'Task 3/Data/test_triplets.txt'

    # generate embedding for each image in the dataset
    """ if(os.path.exists('Task 3/Chris/embeddings.npy') == False):
        generate_embeddings() """

    # load the training data
    X, y = get_data(TRAIN_TRIPLETS)
    
     # Combine X and y for shuffling
    data = list(zip(X, y))
    np.random.shuffle(data)
    # Split the data into training and validation sets (80% train, 20% validation)
    split = int(0.8 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    # Separate features and labels
    X_train, y_train = zip(*train_data)
    X_val, y_val = zip(*val_data)
    
    # Create data loaders for the training data   
    train_loader = create_loader_from_np(np.array(X_train), np.array(y_train), train = True, batch_size=64)
    # Create data loaders for the validation data
    val_loader = create_loader_from_np(np.array(X_val), np.array(y_val), train=False, batch_size=64)
    
    # delete the loaded training data to save memory, as the data loader copies
    del X
    del y
    del data
    del val_data
    del X_train
    del y_train
    del X_val
    del y_val

    # repeat for testing data
    X_test, y_test = get_data(TEST_TRIPLETS, train=False)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)
    del X_test
    del y_test

    # define a model and train it
    model = train_model(train_loader)
    
    val_model = test_model(model, val_loader)
    
    
    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")