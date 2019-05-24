import torch
import glob
import unicodedata
import string
import json
import pickle
import random


all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)
n_training_data = 10

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def convertCuisineName(s):
    if s == "southern_us":
        return "southern"
    elif s == "cajun_creole":
        return "cajun"

    return s

# read data from json file
train_data = json.load(open('../resources/train.json'))
cuisine = set()
ingredients = {}
for t_d in train_data:
    cuisine_name = convertCuisineName(t_d['cuisine']);
    ingre_list = []
    for ingre in t_d['ingredients']:
        ingre_list.append(unicodeToAscii(ingre))
    cuisine.add(cuisine_name);
    if cuisine_name not in ingredients:
        ingredients[cuisine_name] = [];

    for x in range(n_training_data):
        random.shuffle(ingre_list)
        ingredients[cuisine_name].append(ingre_list);

with open('formatted_training_all.json', 'w') as outfile:
    json.dump(ingredients, outfile)

# pickle_out = open("categories.pickle", "wb")
# pickle.dump(cuisine, pickle_out)
# pickle_out.close()

exit();





def findFiles(path): return glob.glob(path)


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, errors='ignore').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('../data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

