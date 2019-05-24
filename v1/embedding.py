import pickle
import numpy as np

def load_glove_model(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r', errors='replace')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model)," words loaded!")
    return model

def get_glove_for_sentence(model, sentence):
    glove_vector = list()
    for word in sentence:
        word = word.lower()
        if word in model.keys():
            glove_vector.append(model[word.lower()])
        else:  # for unknown words, just give all-zero vectors
            glove_vector.append(np.zeros([300,]))
    glove_vector = np.stack(glove_vector, axis=0)
    return glove_vector

if __name__ == '__main__':
    glove_model = load_glove_model('./glove.42B.300d.txt')
    pickle_out = open("glove.pickle", "wb")
    pickle.dump(glove_model, pickle_out)
    pickle_out.close()
    sentence_vec = get_glove_for_sentence(glove_model, ['Hello', 'World'])
