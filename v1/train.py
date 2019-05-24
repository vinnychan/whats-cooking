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
n_epochs = 300000
print_every = 5000
plot_every = 1000
learning_rate = 0.001 # If you set this too high, it might explode. If too low, it might not learn

pickle_in = open("glove.pickle", "rb")
glove_dict = pickle.load(pickle_in)

pickle_in = open("categories.pickle", "rb")
all_categories_set = pickle.load(pickle_in)
all_categories = list(all_categories_set)
train_data = json.load(open('../resources/formatted_training_all.json'))


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

rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
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

all_accuracy = []
training_accuracy = 0

for epoch in range(1, n_epochs + 1):

    category, line, category_tensor, line_tensor = randomTrainingPair()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    guess, guess_i = categoryFromOutput(output)
    if guess == category:
        training_accuracy = training_accuracy + 1

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
            epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        all_accuracy.append(training_accuracy / plot_every)
        current_loss = 0
        training_accuracy = 0



import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
plt.xlabel("number of iteration (1000)")
plt.ylabel("training loss")


plt.figure()
plt.plot(all_accuracy)
plt.xlabel("number of iteration (1000)")
plt.ylabel("training accuracy")

plt.show()

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000
count_correct = 0

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingPair()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1
    if category_i == guess_i:
        count_correct += 1

print('accuracy:')
print(count_correct / n_confusion)

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()

