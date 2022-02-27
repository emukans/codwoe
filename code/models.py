import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import data


class PositionalEncoding(nn.Module):
    """From PyTorch"""

    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LSTMDefmodModel(nn.Module):
    """A LSTM architecture for Definition Modeling."""

    def __init__(
        self, vocab, input_size=256, hidden_size=512, n_head=4, n_layers=1, dropout=0.3, maxlen=256
    ):
        super().__init__()
        self.padding_idx = vocab[data.PAD]
        self.eos_idx = vocab[data.EOS]
        self.maxlen = maxlen
        self.n_head = n_head
        self.vocab = vocab

        self.embedding = nn.Embedding(len(vocab), input_size, padding_idx=self.padding_idx)

        self.hidden_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.v_proj = nn.Linear(hidden_size, len(vocab), bias=False)

        # initializing weights
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            else:  # gain parameters of the layer norm
                nn.init.ones_(param)

    def generate_square_subsequent_mask(self, sz):
        "from Pytorch"
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, sequence, hidden, is_first_iteration):
        embs = self.embedding(sequence)

        if is_first_iteration:
            hidden_proj = (self.hidden_proj(hidden), self.hidden_proj(hidden))
        else:
            hidden_proj = hidden
        out, state = self.lstm(embs, hidden_proj)

        v_dist = F.log_softmax(self.v_proj(out[-1]), dim=1)
        # out = self.ln_f(out)
        # v_dist = self.v_proj(out)

        return v_dist, state

    @staticmethod
    def load(file):
        return torch.load(file)

    def save(self, file):
        torch.save(self, file)


class DefmodModel(nn.Module):
    """A transformer architecture for Definition Modeling."""

    def __init__(
        self, vocab, input_size=256, hidden_size=256, n_head=4, n_layers=1, dropout=0.3, maxlen=256
    ):
        super(DefmodModel, self).__init__()
        self.padding_idx = vocab[data.PAD]
        self.eos_idx = vocab[data.EOS]
        self.maxlen = maxlen
        self.n_head = n_head
        self.vocab = vocab

        self.embedding = nn.Embedding(len(vocab), hidden_size, padding_idx=self.padding_idx)

        self.hidden_proj = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.v_proj = nn.Linear(hidden_size, len(vocab), bias=False)

        # initializing weights
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            else:  # gain parameters of the layer norm
                nn.init.ones_(param)

    def generate_square_subsequent_mask(self, sz):
        "from Pytorch"
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, sequence, hidden, encoder_outputs, is_first_iteration):
        embs = self.embedding(sequence)

        if is_first_iteration:
            hidden_proj = self.hidden_proj(hidden)
        else:
            hidden_proj = hidden
        out, state = self.gru(embs, hidden_proj)

        v_dist = F.log_softmax(self.v_proj(out[-1]), dim=1)

        return v_dist, state

    @staticmethod
    def load(file):
        return torch.load(file)

    def save(self, file):
        torch.save(self, file)


class RevdictModel(nn.Module):
    """A transformer architecture for Definition Modeling."""

    def __init__(
        self, vocab, d_model=256, n_head=4, n_layers=4, dropout=0.3, maxlen=512
    ):
        super(RevdictModel, self).__init__()
        self.d_model = d_model
        self.padding_idx = vocab[data.PAD]
        self.eos_idx = vocab[data.EOS]
        self.maxlen = maxlen

        self.embedding = nn.Embedding(len(vocab), d_model, padding_idx=self.padding_idx)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout=dropout, max_len=maxlen
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dropout=dropout, dim_feedforward=d_model * 2
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.dropout = nn.Dropout(p=dropout)
        self.e_proj = nn.Linear(d_model, d_model)
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            else:  # gain parameters of the layer norm
                nn.init.ones_(param)

    def forward(self, gloss_tensor):
        src_key_padding_mask = gloss_tensor == self.padding_idx
        embs = self.embedding(gloss_tensor)
        src = self.positional_encoding(embs)
        transformer_output = self.dropout(
            self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask.t())
        )
        summed_embs = transformer_output.masked_fill(
            src_key_padding_mask.unsqueeze(-1), 0
        ).sum(dim=0)
        return self.e_proj(F.relu(summed_embs))

    @staticmethod
    def load(file):
        return torch.load(file)

    def save(self, file):
        torch.save(self, file)


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )


#  Implementation of Label smoothing with CrossEntropy and ignore_index
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction="mean", ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index
        )
        return linear_combination(loss / n, nll, self.epsilon)
