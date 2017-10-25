from collections import deque

import torch
from numpy import ndarray
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
from torch import optim
from tqdm import tqdm
import numpy as np
from typing import NamedTuple


Seq2SeqConfig = NamedTuple('Seq2SeqParams', (
    ('message_len', int),
    ('batch_size', int),
    ('context_size', int),
    ('embed_size', int),
    ('use_cuda', bool),
    ('vocab_size', int),
    ('start_token', str),
    ('encoder_layers', int),
    ('learning_rate', float),
    ('teacher_force_ratio', float),
))


def build_model(cfg, start_idx):
    # type: (Seq2SeqConfig, int) -> GruModel
    """ Builds a bomb ass model """
    shared_embedding = build_shared_embedding(cfg)
    encoder = GruEncoder(cfg, shared_embedding, 1)
    decoder = GruDecoder(cfg, shared_embedding, 1)

    if cfg.use_cuda:
        encoder.cuda()
        decoder.cuda()

    return GruModel(cfg, encoder, decoder, shared_embedding, start_idx)


def build_shared_embedding(cfg):
    """ Builds embedding to be used by encoder and decoder """
    # type: Seq2SeqConfig -> nn.Embedding
    return nn.Embedding(cfg.vocab_size, cfg.embed_size)


class GruModel:
    def __init__(self, seq2seq_cfg, encoder, decoder, embedding, start_idx):
        # type: (Seq2SeqConfig, GruEncoder, GruDecoder, nn.Embedding, int) -> None
        self.cfg = seq2seq_cfg
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.start_idx = start_idx

        self.gradient_clip = 5.0
        self.teacher_force_ratio = seq2seq_cfg.teacher_force_ratio
        self.learning_rate = seq2seq_cfg.learning_rate

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.NLLLoss()

    def teacher_should_force(self):
        return random.random() < self.teacher_force_ratio

    def train_epoch(self, train_x, train_y, experiment=None):
        # type: (ndarray, ndarray) -> float
        """ Trains a single epoch. Returns training loss. """
        progress = tqdm(total=len(train_x))
        loss_queue = deque(maxlen=256)
        train_x = train_x.astype('int64')
        train_y = train_y.astype('int64')
        idx_iter = zip(range(0, len(train_x) - self.cfg.batch_size, self.cfg.batch_size),
                       range(self.cfg.batch_size, len(train_x), self.cfg.batch_size))

        for step, (start, end) in enumerate(idx_iter):
            x_batch = train_x[start:end]
            y_batch = train_y[start:end]

            if (len(x_batch) == 0) or (len(y_batch) == 0):
                break

            x_batch = torch.LongTensor(x_batch)
            y_batch = torch.LongTensor(y_batch)

            if self.cfg.use_cuda:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            loss = self._train_inner(
                Variable(x_batch.view(-1, self.cfg.batch_size)),
                Variable(y_batch.view(-1, self.cfg.batch_size)),
            )

            if (experiment is not None) and ((step + 1) % 20 == 0):
                experiment.log_metric('loss', np.mean(loss_queue))

            loss_queue.append(loss)
            progress.set_postfix(loss=np.mean(loss_queue), refresh=False)
            progress.update(self.cfg.batch_size)

        if experiment is not None:
            experiment.log_metric('loss', np.mean(loss_queue))
        return np.mean(loss_queue)

    def _train_inner(self, input_var_batch, target_var_batch):
        # type: (ndarray, ndarray) -> float
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss = 0

        enc_hidden_state = self.encoder.init_hidden()
        encoder_outputs, decoder_hidden = self.encoder(input_var_batch, enc_hidden_state)

        decoder_input = Variable(torch.LongTensor([[self.start_idx]] * self.cfg.batch_size))

        if self.cfg.use_cuda:
            decoder_input = decoder_input.cuda()

        should_use_teacher = self.teacher_should_force()
        for input_idx in range(self.cfg.message_len):

            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            loss += self.loss_fn(decoder_output, target_var_batch[input_idx, :])

            if should_use_teacher:
                decoder_input = target_var_batch[input_idx, :]

            else:
                # Get the highest values and their indexes over axis 1
                top_vals, top_idxs = decoder_output.data.topk(1)
                decoder_input = Variable(top_idxs.squeeze())

        loss.backward()
        nn.utils.clip_grad_norm(self.encoder.parameters(), self.gradient_clip)
        nn.utils.clip_grad_norm(self.decoder.parameters(), self.gradient_clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data.sum() / self.cfg.message_len

    def evaluate(self, test_x, test_y):
        # type: (ndarray, ndarray) -> float
        """ Evaluates model quality on test dataset, returning loss. """


class GruEncoder(nn.Module):
    def __init__(self, seq2seq_params, embedding, n_layers=1):
        # type: (Seq2SeqConfig, nn.Embedding, int) -> None
        super(GruEncoder, self).__init__()
        self.cfg = seq2seq_params
        self.n_layers = seq2seq_params.encoder_layers

        self.embedding = embedding
        self.rnn = nn.GRU(
            input_size=self.cfg.embed_size,
            hidden_size=self.cfg.context_size,
            num_layers=self.n_layers,
        )

    def forward(self, word_idxs, hidden_state):
        embedded = self.embedding(word_idxs) \
            .view(self.cfg.message_len, self.cfg.batch_size, self.cfg.embed_size)

        out, hidden = self.rnn(embedded, hidden_state)
        return out[-1].unsqueeze(0), hidden[-1].unsqueeze(0)

    def init_hidden(self):
        hidden = Variable(torch.randn(self.n_layers, self.cfg.batch_size, self.cfg.context_size))
        return hidden.cuda() if self.cfg.use_cuda else hidden


class GruDecoder(nn.Module):
    def __init__(self, seq2seq_params, embedding, n_layers, dropout_p=0.1):
        # type: (Seq2SeqConfig, nn.Embedding, int, float) -> None
        super(GruDecoder, self).__init__()

        self.cfg = seq2seq_params
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = embedding
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.GRU(
            input_size=self.cfg.embed_size,
            hidden_size=self.cfg.context_size,
            num_layers=self.n_layers,
            dropout=self.dropout_p,
        )
        self.out = nn.Linear(self.cfg.context_size, self.cfg.vocab_size)

    def forward(self, word_idx_slice, last_hidden_state):
        """ Processes a single slice of the minibatch - a single word per row """
        embedded_words = self.embedding(word_idx_slice) \
            .view(1, self.cfg.batch_size, self.cfg.embed_size)
        post_dropout_words = self.dropout(embedded_words)

        output, hidden_state = self.rnn(post_dropout_words, last_hidden_state)
        word_dist = F.log_softmax(self.out(output.squeeze(0)))

        return word_dist, hidden_state
