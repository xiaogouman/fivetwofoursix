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

import pickle

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# input is file name
# output is matrix: row is each text and
def load_train_docs(train_text_file):
    result = []
    with open(train_text_file, "r") as file:
        for line in file:
            result.append(line.split(" "))
    return result

def load_word_embeddings(embeddings_file):
    if os.path.exists('embeddings.pickle'):
        with open('embeddings.pickle', 'rb') as file:
            embedding = pickle.load(file, encoding='utf-8')
        return embedding
    embedding = {}
    with gzip.open(embeddings_file, "rb") as file:
        for l in file:
            line = l.decode().encode('utf-8').decode('utf-8').split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            embedding[word] = vect
    with open('embeddings.pickle', 'wb') as file:
        pickle.dump(embedding, file)
    return embedding

def load_train_label(train_label_file):
    result = []
    with open(train_label_file, "r") as file:
        for line in file:
            result.append(line)
    return result

def train_model(embeddings_file, train_text_file, train_label_file, model_file):
    # write your code here. You can add functions as well.
	# use torch library to save model parameters, hyperparameters, etc. to model_file

    # load data
    train_input = load_train_docs(train_text_file)
    train_label = load_train_label(train_label_file)
    embeddings = load_word_embeddings(embeddings_file)

    first2pairs = {k: embeddings[k] for k in list(embeddings)[:3]}
    for key, value in first2pairs:
        print(key, value)

    # how to use look up table??
    print('Finished...')
		
if __name__ == "__main__":
    # make no changes here
    embeddings_file = sys.argv[1]
    train_text_file = sys.argv[2]
    train_label_file = sys.argv[3]
    model_file = sys.argv[4]
    train_model(embeddings_file, train_text_file, train_label_file, model_file)
