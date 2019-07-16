import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

class charVocabulary(object):
    def __init__(self, token_to_idx=None):
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token 
                                for token, idx in self.token_to_idx.items()}

        self.mask_token = '<mask>'
        self.begin_token = '<begin>'
        self.end_token = '<end>'
        self.unk_token = '<unk>'

        self.mask_idx = self.add_token(self.mask_token)
        self.begin_idx = self.add_token(self.begin_token)
        self.end_idx = self.add_token(self.end_token)
        self.unk_idx = self.add_token(self.unk_token)

    def add_token(self, token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def __len__(self):
        assert len(self.token_to_idx) == len(self.idx_to_token)
        return len(self.token_to_idx)

    def lookup_token(self,token):
        return self.token_to_idx[token]

    def lookup_idx(self,i):
        return self.idx_to_token[i]

    def add_txt(self,path):
        with open(path, 'r') as f:
            fulltext = f.read()
            for c in fulltext:
                if c != '\n':
                    self.add_token(c)
        return None

class charVectorizer(object):
    def __init__(self,vocab):
        self.vocab = vocab

    def vectorize(self, name, max_len=-1):
        """
        max_len is used to know how much to pad
        """
        ind = [self.vocab.begin_idx]
        ind.extend(self.vocab.lookup_token(token) for token in name)
        ind.append(self.vocab.end_idx)
        
        max_len = max(len(ind), max_len)

        x = np.empty(max_len-1, dtype=np.int64)
        x[:len(ind)-1] = ind[:-1]
        x[len(ind)-1:] = self.vocab.mask_idx

        y = np.empty(max_len-1, dtype=np.int64)
        y[:len(ind)-1] = ind[1:]
        y[len(ind)-1:] = self.vocab.mask_idx

        return x,y

class charModel(nn.Module):
    def __init__(self,vocab_size,
                    embedding_dim=10,
                    rnn_hidden_dim=9,
                    padding_idx=0,
                    dropout_p=0.5):
        super(charModel,self).__init__()

        self.emb = nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim=embedding_dim,
                                padding_idx=padding_idx)

        self.rnn = nn.GRU(input_size=embedding_dim,
                            hidden_size=rnn_hidden_dim,
                            batch_first=True)

        self.fc = nn.Linear(in_features=rnn_hidden_dim,
                                out_features=vocab_size)

    def forward(self, x_in, dropout=False, apply_softmax=False):
        x = self.emb(x_in)
        x,_ = self.rnn(x)
        
        batch_size, seq_size, _ = x.shape
        x = x.contiguous().view(batch_size * seq_size, -1)
        x = self.fc(x)

        if dropout:
            x = F.dropout(x,p=self.dropout_p)
        
        if apply_softmax:
            x = F.softmax(x,dim=1)
        
        x = x.view(batch_size, seq_size, -1)
        return x

def run_model(model,vectorizer):
    beginid = vectorizer.vocab.begin_idx
    begintensor = torch.tensor([beginid], dtype=torch.int64).unsqueeze(dim=0)
    ind = [begintensor]
    t = 1
    x_t = ind[1-1]
    h_t = None

    sample_size=25
    for t in range(1,sample_size+1):
        x_t = ind[t-1]
        emb_t = model.emb(x_t)
        rnn_t, h_t = model.rnn(emb_t, h_t)
        pred_vector = model.fc(rnn_t.squeeze(dim=1))
        prob_vector = F.softmax(pred_vector, dim=1)
        winner = torch.multinomial(prob_vector, num_samples=1)
        ind.append(winner)

    s = ""
    for i in range(len(ind)):
        idx = ind[i].item()
        s += vectorizer.vocab.lookup_idx(idx)
        

    i = 0
    while s[i] != '>' and i < len(s):
        i+=1

    out = ""
    j = i+1
    while s[j] != '<' and j < len(s):
        out += s[j]
        j+=1
    
    out = out.capitalize()
    return out

def generate_name(fmodelPath,lmodelPath,vocabPath,rough=False,capitalization=True):

    token_to_idx = pickle.load(open(vocabPath,'rb'))
    vocab = charVocabulary(token_to_idx=token_to_idx)
    vectorizer = charVectorizer(vocab=vocab)

    firstmodel = charModel(vocab_size=len(vocab))
    firstmodel.load_state_dict(torch.load(fmodelPath))

    lastmodel = charModel(vocab_size=len(vocab))
    lastmodel.load_state_dict(torch.load(lmodelPath))

    f = run_model(firstmodel,vectorizer)
    l = run_model(lastmodel,vectorizer)

    name = f+' '+l
    return name

    # beginid = vocab.begin_idx
    # begintensor = torch.tensor([beginid], dtype=torch.int64).unsqueeze(dim=0)
    # ind = [begintensor]
    # t = 1
    # x_t = ind[1-1]
    # h_t = None

    # sample_size=40
    # for t in range(1,sample_size+1):
    #     x_t = ind[t-1]
    #     emb_t = model.emb(x_t)
    #     rnn_t, h_t = model.rnn(emb_t, h_t)
    #     pred_vector = model.fc(rnn_t.squeeze(dim=1))
    #     prob_vector = F.softmax(pred_vector, dim=1)
    #     winner = torch.multinomial(prob_vector, num_samples=1)
    #     ind.append(winner)

    # s = ""
    # for i in range(len(ind)):
    #     idx = ind[i].item()
    #     s += vectorizer.vocab.lookup_idx(idx)
        
    # if rough:
    #     return s
    # else:
    #     i = 0
    #     while s[i] != '>' and i < len(s):
    #         i+=1

    #     out = ""
    #     j = i+1
    #     while s[j] != '<' and j < len(s):
    #         out += s[j]
    #         j+=1
        
    #     if capitalization:
    #         out = out.capitalize()
    #     return out