from model import *
from data import *
from embedding import *
import sys
import torch

pickle_in = open("glove.pickle", "rb")
glove_dict = pickle.load(pickle_in)


pickle_in = open("categories.pickle", "rb")
all_categories_set = pickle.load(pickle_in)
all_categories = list(all_categories_set)

rnn = torch.load('char-rnn-classification.pt')


def normalizeTrainingData(trainingArray):
    normalizedArray = []
    for trainData in trainingArray:
        dataSplit = trainData.split()
        for data in dataSplit:
            normalizedArray.append(data.strip())

    return normalizedArray

def randomTrainingPair(line):
    line_tensor = Variable(torch.FloatTensor(get_glove_for_sentence(glove_dict, normalizeTrainingData(line))))
    return line_tensor

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

#def predict(line, n_predictions=3):
def predict(line, n_predictions=3):
    line_tensor = randomTrainingPair(line)
    output = evaluate(line_tensor)

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions

if __name__ == '__main__':
    predict(sys.argv[1])
