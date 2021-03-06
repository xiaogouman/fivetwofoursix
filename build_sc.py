# python3.5 build_sc.py <pretrained_vectors_gzipped_file_absolute_path> <train_text_path> <train_label_path> <model_file_path>

import os
import math
import sys
import gzip
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils import data
import csv


EMBEDDING_DIM = 300
MAX_LENGTH = 300
EPOCHS = 20
LR = 0.001
BATCH_SIZE = 50


print(os.getcwd())
file_path = '../weight_matrix.pkl'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)


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
        # class = 1 => label = 0, class = 2 => label = 1
        if torch.cuda.is_available():
            doc = torch.tensor(self.X[index]).long().cuda()
            label = torch.tensor(self.Y[index] - 1).long().cuda()
        else:
            doc = torch.tensor(self.X[index]).long()
            label = torch.tensor(self.Y[index]-1).long()
        if self.transform is not None:
            doc = self.transform(doc)
        return doc, label


class ConvNet(nn.Module):

    def __init__(self, embeddings, num_classes=2):
        super(ConvNet, self).__init__()
        self.embedding = nn.Embedding(list(embeddings.size())[0], EMBEDDING_DIM, _weight=embeddings, padding_idx=0)
        # self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embeddings))
        self.conv2 = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=256, kernel_size=2, stride=1, padding=2-1)
        self.conv3 = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=128, kernel_size=3, stride=1, padding=3-1)
        self.conv4 = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=96, kernel_size=4, stride=1, padding=4-1)
        # self.conv5 = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=32, kernel_size=5, stride=1, padding=5-1)
        # self.conv5 = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=N_FILTERS, kernel_size=5, stride=1, padding=5-1)
        self.fc = nn.Linear(256+128+96, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out = out.permute(0, 2, 1)
        out2 = F.relu(self.conv2(out)).max(dim=2)[0]
        out3 = F.relu(self.conv3(out)).max(dim=2)[0]
        out4 = F.relu(self.conv4(out)).max(dim=2)[0]
        # out5 = F.relu(self.conv5(out)).max(dim=2)[0]
        out = torch.cat((out2, out3, out4), dim=1)
        # out = F.dropout(out, p=0.5)
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out


# input:  is file name
# output: matrix that each row is [w1, w2, w3 ...] for each doc, and set of all vocabulary
def load_docs(docs_file):
    result = []
    with open(docs_file, "r") as file:
        for line in file.readlines():
            # remove punctuations
            line = re.sub(r'[^\w\s]', '', line)
            words = [w.lower() for w in line.strip().split(' ') if w != '']
            result.append(words)
    return result


def load_label(filename):
    result = []
    with open(filename, "r") as file:
        for line in file:
            result.append(int(line))
    return result


def gen_word_index(docs, model_file):
    max_length = 0
    target_vocab = set()
    word_idx = {}
    index = 1  # index = 0 is used for padding
    for doc in docs:
        target_vocab.update(doc)
        if len(doc) > max_length:
            max_length = len(docs)
    print("max number of words: ", max_length)
    for word in target_vocab:
        word_idx[word] = index
        index = index + 1

    # save word index
    print('saving word index')
    with open('{}/word_idx.csv'.format(model_file), 'w', newline='') as csvfile:
        w = csv.writer(csvfile)
        for key, val in word_idx.items():
            w.writerow([key, val])
    return word_idx


def word_to_idx_inputs(docs, word_index):
    # transform from word to index
    index_input = []
    for doc in docs:
        idxs = [word_index[word] if word in word_index else 0 for word in doc]
        # make input length = MAX_LENGTH
        padded_idxs = []
        for i in range(MAX_LENGTH):
            if i < len(idxs):
                padded_idxs.append(idxs[i])
            else:
                padded_idxs.append(0)
        index_input.append(padded_idxs)
    return index_input


def load_word_embeddings(embeddings_file, word2idx):
    print('start reading from embedding file')
    with gzip.open(embeddings_file, 'rt', encoding='utf-8') as f:
        embeddings = torch.rand(len(word2idx)+1, EMBEDDING_DIM) * 0.5 - 0.25
        embeddings[0] = torch.zeros((EMBEDDING_DIM,))
        # embeddings = torch.zeros(len(word2idx)+1, EMBEDDING_DIM)
        for line in f:
            line = line.strip()
            first_space_pos = line.find(' ', 1)
            word = line[:first_space_pos]
            if word in word2idx:
                idx = word2idx[word]
                emb_str = line[first_space_pos + 1:].strip()
                emb = [float(t) for t in emb_str.split(' ')]
                embeddings[idx] = torch.tensor(emb)
    print('finish reading from embedding file')
    return embeddings


def train_model(embeddings_file, train_text_file, train_label_file, model_file):
    # write your code here. You can add functions as well.
	# use torch library to save model parameters, hyperparameters, etc. to model_file

    # load data
    train_docs = load_docs(train_text_file)
    train_label = load_label(train_label_file)
    word_index = gen_word_index(train_docs, model_file)
    train_input = word_to_idx_inputs(train_docs, word_index)
    train_dataset = DatasetDocs(train_input, train_label)

    test_docs = load_docs('docs.test')
    test_label = load_label('classes.test')
    test_input = word_to_idx_inputs(test_docs, word_index)
    test_dataset = DatasetDocs(test_input, test_label)

    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # define model
    embeddings = load_word_embeddings(embeddings_file, word_index)
    model = ConvNet(embeddings).to(device)
    print(model)


    ######## training ############
    print('training start')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        # switch model to train mode
        model.train()
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pas
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # switch model to evaluation mode
        model.eval()

        # calculate accuracy on validation set
        n_val_correct, val_loss = 0, 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                n_val_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
                val_loss = criterion(outputs, labels).item()
        val_acc = n_val_correct / (len(val_dataloader) * BATCH_SIZE)

        n_train_correct, train_loss = 0, 0
        with torch.no_grad():
            for inputs, labels in train_dataloader:
                outputs = model(inputs)
                n_train_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
                train_loss = criterion(outputs, labels).item()
        train_acc = n_train_correct / (len(train_dataloader) * BATCH_SIZE)

        print('Epoch [{}/{}], Train acc: {:.4f}, Train Loss: {:.4f}, Val Acc: {:.4f}, Val Loss: {:.4f}'
              .format(epoch + 1, EPOCHS, train_acc, train_loss, val_acc, val_loss))

    torch.save(model, '{}/model.pth'.format(model_file))
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    embeddings_file = sys.argv[1] if len(sys.argv) > 3 else "/Users/xiaogouman/Documents/masters/CS5246/Assignment/assignment_1/vectors.txt.gz"
    train_text_file = sys.argv[2] if len(sys.argv) > 3 else "docs.train"
    train_label_file = sys.argv[3] if len(sys.argv) > 3 else "classes.train"
    model_file = sys.argv[4] if len(sys.argv) > 3 else "model_file"

    if not os.path.exists(model_file):
        os.makedirs(model_file)
    # train_model(embeddings_file, train_text_file, train_label_file, model_file)

    from _datetime import datetime
    for BATCH_SIZE in [100]:
        for MAX_LENGTH in [500, 1000]:
            for LR in [0.001, 0.0005]:
                print('=========== start of train ============')
                print('BATCH_SIZE: ', BATCH_SIZE)
                print('MAX_LENGTH: ', MAX_LENGTH)
                print('LR: ', LR)
                time_before = datetime.now()
                train_model(embeddings_file, train_text_file, train_label_file, model_file)
                time_taken = (datetime.now()-time_before).total_seconds() / 60
                print('time taken: {:.2f}'.format(time_taken))
                print('=========== end of train ============')
