# python3.5 run_sc.py <test_file_path> <model_file_path> <output_file_path>

import os
import math
import sys
import torch
import re
import csv
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 300
MAX_LENGTH = 1000
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ConvNet(nn.Module):

    def __init__(self, embeddings, num_classes=2):
        super(ConvNet, self).__init__()
        self.embedding = nn.Embedding(list(embeddings.size())[0], EMBEDDING_DIM, _weight=embeddings, padding_idx=0)
        self.conv2 = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=256, kernel_size=2, stride=1, padding=2-1)
        self.conv3 = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=128, kernel_size=3, stride=1, padding=3-1)
        self.conv4 = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=96, kernel_size=4, stride=1, padding=4-1)
        self.fc = nn.Linear(256+128+96, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out = out.permute(0, 2, 1)
        out2 = F.relu(self.conv2(out)).max(dim=2)[0]
        out3 = F.relu(self.conv3(out)).max(dim=2)[0]
        out4 = F.relu(self.conv4(out)).max(dim=2)[0]
        out = torch.cat((out2, out3, out4), dim=1)
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


def load_word_idx(model_file):
    word_idx = {}
    with open('{}/word_idx.csv'.format(model_file)) as csvreader:
        r = csv.reader(csvreader, delimiter=',')
        for row in r:
            word_idx[row[0]] = int(row[1])
    return word_idx


def test_model(test_text_file, model_file, out_file):
    # write your code here. You can add functions as well.
    # use torch library to load model_file

    # read test data
    test_docs = load_docs(test_text_file)

    # load word index
    word_idx = load_word_idx(model_file)

    # transfer words to index
    test_input = word_to_idx_inputs(test_docs, word_idx)

    # load model
    model = torch.load('{}/model.pth'.format(model_file)).to(device)
    model.eval()

    # get output
    test_input = torch.tensor(test_input).cuda() if torch.cuda.is_available() else torch.tensor(test_input)
    outputs = model(test_input)

    # write to file
    labels = []
    for label in torch.max(outputs, 1)[1]:
        labels.append(label.item()+1)

    with open(out_file, 'w') as file:
        for i, label in enumerate(labels):
            if i == len(labels) - 1:
                file.write(str(label))
            else:
                file.write(str(label)+'\n')

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_text_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    test_model(test_text_file, model_file, out_file)
