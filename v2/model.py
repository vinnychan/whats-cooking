import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.input2output = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.input2hidden(combined)
        output = self.input2output(combined)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_layer = nn.LSTM(input_size, hidden_size, 1, bidirectional=True)
        self.hidden_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, (hidden, _) = self.lstm_layer(input, hidden)
        output = self.hidden_layer(output)
        return output, (hidden, _)

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))