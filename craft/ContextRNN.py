from torch import nn

class ContextEncoderRNN(nn.Module):
    """This module represents the context encoder component of CRAFT, responsible for creating an order-sensitive vector representation of conversation context"""
    def __init__(self, hidden_size, n_layers=1, dropout=0):
        super(ContextEncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        # only unidirectional GRU for context encoding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first = True,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=False)

    def forward(self, input_seq, hidden=None):

        outputs, hidden = self.gru(input_seq, hidden)

        return outputs, hidden

class SingleTargetClf(nn.Module):
    """This module represents the CRAFT classifier head, which takes the context encoding and uses it to make a forecast"""
    def __init__(self, hidden_size, dropout=0.1):
        super(SingleTargetClf, self).__init__()

        self.hidden_size = hidden_size

        # initialize classifier
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer1_act = nn.LeakyReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer2_act = nn.LeakyReLU()
        self.clf = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, encoder_hidden):
        hidden = encoder_hidden[-1,:,:]
        # forward pass through hidden layers
        hidden = hidden.squeeze()
        layer1_out = self.layer1_act(self.layer1(self.dropout(hidden)))
        layer2_out = self.layer2_act(self.layer2(self.dropout(layer1_out)))
        # compute and return logits
        logits = self.clf(self.dropout(layer2_out)).squeeze()
        return logits