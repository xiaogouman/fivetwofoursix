# python3.5 build_sc.py <pretrained_vectors_gzipped_file_absolute_path> <train_text_path> <train_label_path> <model_file_path>

import os
import math
import sys
import gzip
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import pickle

EMBEDDING_DIM = 300
INPUT_DIM = 1500
KERNEL_SIZES = [3]
epochs = 1
running_loss = 0
print_every = 10
learning_rate = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# embedding in pytorch -> https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
# cnn in pytorch -> https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L35-L56
# pretrained embedding in pytorch -> https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
# text classification using keras -> https://realpython.com/python-keras-text-classification/#what-is-a-word-embedding
# pytorch basics -> https://cs230-stanford.github.io/pytorch-getting-started.html

class DatasetDocs(Dataset):

    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        doc = torch.Tensor(self.X[index])
        label = torch.Tensor(self.Y[index])

        if self.transform is not None:
            doc = self.transform(doc)

        return doc, label


class ConvNet(nn.Module):
    def create_embedding_layer(self, weight_matrix):
        # layer = nn.Embedding(len(weight_matrix), len(weight_matrix[0]), padding_idx=0)
        # weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
        layer = nn.Embedding.from_pretrained(torch.FloatTensor(weight_matrix))

        # weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
        # layer = nn.Embedding.from_pretrained(weight)
        return layer

    def create_conv_layers(self, input_dim, kernal_sizes, n_filters=32):
        layers = []
        for kernal_size in kernal_sizes:
            layer = nn.Conv1d(input_dim, n_filters, kernel_size=kernal_size, stride=1, padding=kernal_size-1)
            layers.append(layer)
            self.max_pooled_vev_dim = self.max_pooled_vev_dim + n_filters
        return layers

    def __init__(self, weight_matrix, kernal_sizes, num_classes=2):
        super(ConvNet, self).__init__()
        self.max_pooled_vev_dim = 0
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weight_matrix))
        self.conv_layers = self.create_conv_layers(INPUT_DIM, kernal_sizes)
        self.fc = nn.Linear(self.max_pooled_vev_dim, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        cat_output = torch.Tensor()
        for layer in self.conv_layers:
            layer_output = layer(out)
            layer_output = F.relu(layer_output)
            layer_output = F.max_pool1d(torch.t(layer_output))
            cat_output = torch.cat((cat_output, layer_output), 1)

        out = self.fc(cat_output)
        out = F.softmax(out)
        return out



# input:  is file name
# output: matrix that each row is [w1, w2, w3 ...] for each doc, and set of all vocabulary
def load_train_docs(train_text_file):
    result = []
    max_length = 0
    target_vocab = set()
    word_index = dict()
    index = 1 # index = 0 is used for padding
    with open(train_text_file, "r") as file:
        for line in file:
            # TODO: tokenize words
            words = line.split(" ")
            result.append(words)
            target_vocab.update(words)
            if len(words) > max_length:
                max_length = len(words)
    print ("max number of words: ", max_length)
    for word in target_vocab:
        word_index[word] = index
        index = index + 1

    index_input = []
    for doc in result:
        idxs = [word_index[word] for word in doc]
        index_input.append(idxs)
    return index_input, word_index

def load_word_embeddings(embeddings_file, word_index):
    file_path = '../weight_matrix.npy'
    if os.path.exists(file_path):
        print('start loading weight_matrix')
        weight_matrix = np.load(file_path)
        print('finish loading weight_matrix')
        return weight_matrix

    print('start reading from embedding file')
    embedding = {}
    weight_matrix = np.zeros((len(word_index), EMBEDDING_DIM))
    with gzip.open(embeddings_file, "rb") as file:
        for l in file:
            line = l.decode().encode('utf-8').decode('utf-8').split()
            word, vect = line[0], np.array(line[1:]).astype(np.float)
            embedding[word] = vect
    print('finish reading from embedding file')

    # if pretrained matrix does not exist, generate random weight
    found = 0
    for word, idx in word_index.items():
        if word in embedding:
            found = found + 1
            weight_matrix[idx] = embedding[word]
        else:
            weight_matrix[idx] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))
    print('total vocab: ', len(word_index), 'found: ', found)

    print('start saving weight_matrix')
    np.save(file_path, weight_matrix)
    print('finish saving weight_matrix')
    return weight_matrix

def load_train_label(train_label_file):
    result = []
    with open(train_label_file, "r") as file:
        for line in file:
            result.append(int(line))
    return result

from torch.utils import data

def split_data(dataset):
    n = len(dataset)
    n_train = int(n*0.8)
    n_val = n - n_train
    train_set, val_set = data.random_split(dataset, (n_train, n_val))
    return data.DataLoader(train_set), data.DataLoader(val_set)


from torchsummary import summary

def train_model(embeddings_file, train_text_file, train_label_file, model_file):
    # write your code here. You can add functions as well.
	# use torch library to save model parameters, hyperparameters, etc. to model_file

    # load data
    train_input, word_index = load_train_docs(train_text_file)
    train_label = load_train_label(train_label_file)
    weight_matrix = load_word_embeddings(embeddings_file, word_index)

    model = ConvNet(weight_matrix,KERNEL_SIZES).to(device)
    print(summary(model, (1000, 2000, 300)))

    train_dataset = DatasetDocs(train_input, train_label)
    train_dataloader, val_dataloader = split_data(train_dataset)


    ######## training ############
    train_losses, test_losses = [], []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps=0
    total_steps=len(train_input)
    for epoch in range(epochs):
        for inputs, labels in train_dataloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (steps) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, steps, total_steps, loss.item()))

    torch.save(model.state_dict(), 'model.ckpt')


    print('Finished...')
		
if __name__ == "__main__":
    # make no changes here
    embeddings_file = sys.argv[1]
    train_text_file = sys.argv[2]
    train_label_file = sys.argv[3]
    model_file = sys.argv[4]
    train_model(embeddings_file, train_text_file, train_label_file, model_file)
