import requests
import nltk
import unicodedata
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
"""
Due to the specialized nature of the forecasting task, we define the UtteranceModel to be able to tokenize the utterances.
"""
class Voc:
    """A class for representing the vocabulary used by a CRAFT model"""

    def __init__(self, name, word2index=None, index2word=None):
        # Default word tokens
        self.PAD_token = 0  # Used for padding short sentences
        self.SOS_token = 1  # Start-of-sentence token
        self.EOS_token = 2  # End-of-sentence token
        self.UNK_token = 3  # Unknown word token

        self.name = name
        self.trimmed = False if not word2index else True # if a precomputed vocab is specified assume the user wants to use it as-is
        self.word2index = word2index if word2index else {"UNK": self.UNK_token}
        self.word2count = {}
        self.index2word = index2word if index2word else {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS", self.UNK_token: "UNK"}
        self.num_words = 4 if not index2word else len(index2word)  # Count SOS, EOS, PAD, UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {"UNK": self.UNK_token}
        self.word2count = {}
        self.index2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS", self.UNK_token: "UNK"}
        self.num_words = 4 # Count default tokens
        for word in keep_words:
            self.addWord(word)
class Tokenizer:
    def __init__(self):
        self.tokenizer = nltk.tokenize.RegexpTokenizer(pattern=r'\w+|[^\w\s]')
        WORD2INDEX_URL = "http://zissou.infosci.cornell.edu/convokit/models/craft_wikiconv/word2index.json"
        INDEX2WORD_URL = "http://zissou.infosci.cornell.edu/convokit/models/craft_wikiconv/index2word.json"
        self.vocab = self.loadPrecomputedVoc("wikiconv", WORD2INDEX_URL, INDEX2WORD_URL)
        self.vocab_size = self.vocab.num_words
    # Create a Voc object from precomputed data structures
    def loadPrecomputedVoc(self, corpus_name, word2index_url, index2word_url):
        # load the word-to-index lookup map
        r = requests.get(word2index_url)
        word2index = r.json()
        # load the index-to-word lookup map
        r = requests.get(index2word_url)
        index2word = r.json()
        return Voc(corpus_name, word2index, index2word)
    def unicodeToAscii(self, utterance):
        return ''.join(
            c for c in unicodedata.normalize('NFD', utterance)
            if unicodedata.category(c) != 'Mn')
    def tokenize(self, utterance):
        # simplify the problem space by considering only ASCII data
        cleaned_text = self.unicodeToAscii(utterance.lower())
        # if the resulting string is empty, nothing else to do
        if not cleaned_text.strip():
            return []
        return self.tokenizer.tokenize(cleaned_text)
    
    def forward(self, utterance):
        tokens = self.tokenize(utterance)
        inputs = []
        for token in tokens:
            if token in self.vocab.word2index:
                inputs.append(self.vocab.word2index[token])
            else:
                inputs.append(self.vocab.UNK_token)
        inputs.append(self.vocab.EOS_token)
        return inputs
    
class EncoderRNN(nn.Module):
    """This module represents the utterance encoder component of CRAFT, responsible for creating vector representations of utterances"""
    def __init__(self, device, hidden_size, embedding_dim, max_utterance_len, batch_size=256, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.tokenizer = Tokenizer()
        self.max_utterance_len = max_utterance_len
        self.batch_size = batch_size
        self.device = device
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, embedding_dim).to(device)

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(embedding_dim, hidden_size, n_layers, batch_first = True,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True).to(device)
    def tokenize(self, utterances):
        inputs = [self.tokenizer.forward(utterance) for utterance in utterances]
        features = np.zeros((len(inputs), self.max_utterance_len), dtype=int)

        # for each review, I grab that review and 
        for i, row in enumerate(inputs):
            features[i, -min(len(row), self.max_utterance_len):] = np.array(row)[:self.max_utterance_len]

        def batch(features, batch_size):
            for i in range(0, len(features), batch_size):
                yield torch.from_numpy(features[i:min(i+batch_size, len(features))]).long().to(self.device)

        return batch(features, self.batch_size)
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = [weight.new(self.n_layers, batch_size, self.hidden_size).zero_() for _ in range(self.n_layers)]
        return torch.cat(hidden, dim=0).to(self.device)
    def forward(self, utterances, hidden=None):
        dataloader = self.tokenize(utterances)
        outputs = []
        hiddens = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            hidden = self.init_hidden(inputs.shape[0])
            embedded = self.embedding(inputs)
            output, hidden = self.gru(embedded, hidden)
            outputs.append(output.to('cpu').detach())
            hiddens.append(hidden.to('cpu').detach())
        outputs = torch.cat(outputs, dim=0)
        hiddens = torch.cat(hiddens, dim=1)
        return outputs, hiddens
    

def load_Utterance_pretrain(device):
    hidden_size = 500
    encoder_n_layers = 2
    dropout = 0.1
    # MODEL_URL = "http://zissou.infosci.cornell.edu/convokit/models/craft_wikiconv/craft_pretrained.tar"
    # print("Loading saved parameters...")
    # if not os.path.isfile("pretrained_model.tar"):
    #     print("\tDownloading pre-trained CRAFT...")
    #     wget.download(MODEL_URL)
    #     # urlretrieve(MODEL_URL, "craft_pretrained.tar")
    #     print("\t...Done!")
    checkpoint = torch.load("craft_pretrained.tar")
    encoder = EncoderRNN(device=device, hidden_size=hidden_size, embedding_dim=hidden_size, max_utterance_len = 100,
                          n_layers=encoder_n_layers, dropout=dropout)
    encoder_sd = checkpoint['en']
    embedding_sd = checkpoint['embedding']
    encoder.embedding.load_state_dict(embedding_sd)
    encoder.load_state_dict(encoder_sd)
    return encoder