# import torch
# from data import *
from model import *
import random
import time
import math
import pickle
import torch
import glob
import unicodedata
import string
import json
from embedding import *

n_hidden = 300
n_epochs = 200000
print_every = 5000
plot_every = 1000
learning_rate = 0.0008 # If you set this too high, it might explode. If too low, it might not learn

pickle_in = open("glove.pickle", "rb")
glove_dict = pickle.load(pickle_in)

pickle_in = open("categories.pickle", "rb")
all_categories_set = pickle.load(pickle_in)
all_categories = list(all_categories_set)
train_data = json.load(open('../resources/formatted_data.json'))


all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)
n_categories = len(all_categories)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )



# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    return tensor


def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def normalizeTrainingData(trainingArray):
    normalizedArray = []
    for trainData in trainingArray:
        dataSplit = trainData.split()
        for data in dataSplit:
            normalizedArray.append(data.strip())

    return normalizedArray

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(train_data[category])

    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = Variable(torch.FloatTensor(get_glove_for_sentence(glove_dict, normalizeTrainingData(line))))
    return category, line, category_tensor, line_tensor

model_type = 'LSTM'

if model_type == 'LSTM':
    model = LSTM(300, n_hidden, n_categories)
else:
    model = RNN(300, n_hidden, n_categories)


#rnn = torch.load("char-rnn-classification-3.pt")
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()



def train(category_tensor, line_tensor):
    if model_type == 'LSTM':
        hidden = model.initHidden().reshape(1, 1, 300)
        c_hidden = model.initHidden().reshape(1, 1, 300)
        optimizer.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, (hidden, c_hidden) = model(line_tensor[i].reshape((1, 1, 300)), (hidden, c_hidden))
    else:
        hidden = model.initHidden()
        optimizer.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[i], hidden)


    loss = criterion(output.reshape((1, 20)), category_tensor)
    loss.backward()


    optimizer.step()

    return output, loss.item()

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = randomTrainingPair()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn, 'char-rnn-classification-5.pt')
