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
from torch.utils import data


EMBEDDING_DIM = 300
N_FILTERS = 3
MAX_LENGTH = 50
KERNEL_SIZES = [3, 4]
EPOCHS = 100
LR = 0.01
BATCH_SIZE = 10

file_path = '../weight_matrix.npy'
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
        doc = torch.from_numpy(np.asarray(self.X[index])).long()
        # class = 1 => label = 0, class = 2 => label = 1
        label = torch.from_numpy(np.asarray(self.Y[index]-1)).long()
        if self.transform is not None:
            doc = self.transform(doc)
        return doc, label


class ConvNet(nn.Module):
    def create_conv_layers(self, kernal_sizes):
        layers = []
        for kernal_size in kernal_sizes:
            layer = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=N_FILTERS, kernel_size=kernal_size, stride=1, padding=kernal_size-1)
            layers.append(layer)
        return layers

    def __init__(self, embeddings, kernal_sizes, num_classes=2):
        super(ConvNet, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embeddings))
        self.conv_layers = self.create_conv_layers(kernal_sizes)
        self.fc = nn.Linear(len(KERNEL_SIZES)*N_FILTERS, num_classes)

    def forward(self, x):
        # print('input size', x.size())
        out = self.embedding(x)
        # print('embedding out size', out.size())
        out = out.permute(0, 2, 1)
        # print ('permute out size', out.size())
        out = [F.relu(conv(out)).max(2)[0] for conv in self.conv_layers]
        out = torch.cat(out, 1)
        # print('conv out size', out.size())
        out = self.fc(out)
        # print('fc out size', out.size())
        out = F.softmax(out, dim=1)
        # print('final output', out)
        return out


# input:  is file name
# output: matrix that each row is [w1, w2, w3 ...] for each doc, and set of all vocabulary
def load_train_docs(train_text_file):
    result = []
    max_length = 0
    target_vocab = set()
    word_index = dict()
    index = 1  # index = 0 is used for padding
    with open(train_text_file, "r") as file:
        for line in file.readlines():
            # TODO: tokenize words
            words = [w.lower() for w in line.strip().split(' ')[2:]]
            result.append(words)
            target_vocab.update(words)
            if len(words) > max_length:
                max_length = len(words)
    print("max number of words: ", max_length)
    for word in target_vocab:
        word_index[word] = index
        index = index + 1

    # transform from word to index
    index_input = []
    for doc in result:
        idxs = [word_index[word] for word in doc]
        # make input length = MAX_LENGTH
        idxs = np.pad(idxs, (0, MAX_LENGTH-len(idxs)), 'constant') if len(idxs) < MAX_LENGTH else idxs[:MAX_LENGTH]
        index_input.append(idxs)
    return index_input, word_index


def load_word_embeddings(embeddings_file, word2idx):
    if os.path.exists(file_path):
        print('start loading weight_matrix')
        embeddings = np.load(file_path)
        print('finish loading weight_matrix')
        return embeddings

    print('start reading from embedding file')
    with gzip.open(embeddings_file, 'rt', encoding='utf-8') as f:
        embeddings = torch.rand(len(word2idx)+1, EMBEDDING_DIM) * 0.5 - 0.25
        embeddings[0] = torch.zeros((EMBEDDING_DIM,))
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

    # save weight matrix
    print('start saving weight_matrix')
    np.save(file_path, embeddings)
    print('finish saving weight_matrix')

    # save word index
    print('saving word index')
    w = csv.writer(open('word_idx.csv', 'w'))
    for key, val in word2idx.items():
        w.writerow([key, val])
    return embeddings


def load_train_label(train_label_file):
    result = []
    with open(train_label_file, "r") as file:
        for line in file:
            result.append(int(line))
    return result


def split_data(dataset):
    n = len(dataset)
    n_train = int(n*0.8)
    n_val = n - n_train
    train_set, val_set = data.random_split(dataset, (n_train, n_val))
    return data.DataLoader(train_set, batch_size=BATCH_SIZE), data.DataLoader(val_set, batch_size=BATCH_SIZE)


from torchsummary import summary
import csv

def train_model(embeddings_file, train_text_file, train_label_file, model_file):
    # write your code here. You can add functions as well.
	# use torch library to save model parameters, hyperparameters, etc. to model_file

    # load data
    train_input, word_index = load_train_docs(train_text_file)
    train_label = load_train_label(train_label_file)
    dataset = DatasetDocs(train_input, train_label)
    train_dataloader, val_dataloader = split_data(dataset)
    print('length; ', len(dataset))

    # define model
    embeddings = load_word_embeddings(embeddings_file, word_index)
    # print(embeddings)
    model = ConvNet(embeddings, KERNEL_SIZES).to(device)
    if torch.cuda.is_available():
        model.cude()

    print(model)

    # print(summary(model, (1, 2000, 300)))

    ######## training ############
    print('training start')
    train_losses, test_losses = [], []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    total_steps = len(train_dataloader)
    for epoch in range(EPOCHS):
        for inputs, labels in train_dataloader:
            # print("inputs: ", inputs)
            # print("labels: ", labels)
            model.train()
            optimizer.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)

            # forward pas
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if steps % total_steps/20 == 0:

                # checkpoint model periodically
                # if steps % args.save_every == 0:
                #     snapshot_prefix = os.path.join(args.save_path, 'snapshot')
                #     snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc,
                #                                                                                         loss.item(),
                #                                                                                         iterations)
                #     torch.save(model, snapshot_path)
                #     for f in glob.glob(snapshot_prefix + '*'):
                #         if f != snapshot_path:
                #             os.remove(f)c

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

        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Acc: {:.2f}, Val Loss: {:.4f}'
              .format(epoch + 1, EPOCHS, loss.item(), val_acc, val_loss))

    torch.save(model.state_dict(), 'model.ckpt')
    print('Finished...')
		
if __name__ == "__main__":
    # make no changes here
    embeddings_file = sys.argv[1] if len(sys.argv) > 3 else "/Users/xiaogouman/Documents/masters/CS5246/Assignment/assignment_1/vectors.txt.gz"
    train_text_file = sys.argv[2] if len(sys.argv) > 3 else "docs.train"
    train_label_file = sys.argv[3] if len(sys.argv) > 3 else "classes.train"
    model_file = sys.argv[4] if len(sys.argv) > 3 else "model_file"

    # embeddings_file = "/Users/xiaogouman/Documents/masters/CS5246/Assignment/assignment_1/vectors.txt.gz"
    # train_text_file = "docs.train"
    # train_label_file = "classes.train"
    # model_file = "model_file"
    train_model(embeddings_file, train_text_file, train_label_file, model_file)
