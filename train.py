
import os
USE_COMET = os.environ.get('COMET') == 'TRUE'

if USE_COMET:
    from comet_ml import Experiment
from bpe import Encoder
from gru_model import Seq2SeqConfig, build_model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize
import logging
import toolz


S2S_PARAMS = Seq2SeqConfig(
    message_len=30,
    batch_size=int(os.environ.get('BATCH_SIZE', '32')),
    context_size=200,
    embed_size=200,
    use_cuda=True,
    vocab_size=2**13,
    start_token='__start__',
    encoder_layers=int(os.environ.get('ENCODER_LAYERS', '2')),
    learning_rate=float(os.environ.get('LEARNING_RATE', '0.005')),
)

LIMIT = int(os.environ.get('LIMIT', '10000000'))


def train():
    x_lines = [*toolz.take(LIMIT, open('data/x.txt').read().split('\n'))]
    y_lines = [*toolz.take(LIMIT, open('data/y.txt').read().split('\n'))]

    encoder = encoder_for_lines(S2S_PARAMS, x_lines + y_lines)

    try:
        start_idx = encoder.word_vocab[S2S_PARAMS.start_token]
    except AttributeError:
        start_idx = int(encoder.vocabulary_[S2S_PARAMS.start_token])

    model = build_model(S2S_PARAMS, start_idx)

    x = encode_data(encoder, x_lines, is_input=True)
    y = encode_data(encoder, y_lines, is_input=False)

    print(x.shape, y.shape)

    x = x[:S2S_PARAMS.batch_size*int(len(x)/S2S_PARAMS.batch_size)]
    y = y[:S2S_PARAMS.batch_size*int(len(y)/S2S_PARAMS.batch_size)]

    if USE_COMET:
        experiment = Experiment(api_key="DQqhNiimkjP0gK6c8iGz9orzL", log_code=True)
        experiment.log_multiple_params(S2S_PARAMS._asdict())
        for idx in range(1000):
            print("Training in epoch " + str(idx))
            model.train_epoch(x, y, experiment=experiment)
            experiment.log_epoch_end(idx)
    else:
        for idx in range(1000):
            print("Training in epoch " + str(idx))
            model.train_epoch(x, y)


def sklearn_encoder_for_data(cfg: Seq2SeqConfig, lines):
    logging.info("Fitting sklearn count vectorizer...")
    cv = CountVectorizer(tokenizer=casual_tokenize, max_features=cfg.vocab_size - 1)
    cv.fit(lines + [cfg.start_token] * 100000)
    return cv


def bpe_encoder_for_lines(cfg: Seq2SeqConfig, lines) -> Encoder:
    """ Calculate BPE encoder for provided lines of text """
    encoder = Encoder(vocab_size=cfg.vocab_size, required_tokens=[cfg.start_token])
    encoder.fit(lines)
    encoder.save('latest_encoder.json')
    return encoder


def sklearn_encode_data(encoder, text, is_input=False):
    encoded = np.zeros((len(text), 30), dtype='int32')
    unk = encoder.max_features

    for row_idx, line in enumerate(text):
        for col_idx, token in enumerate(toolz.take(30, encoder.tokenizer(line))):
            encoded[row_idx][col_idx] = encoder.vocabulary_.get(token, unk)

    return encoded.astype('int32')

def bpe_encode_data(encoder, text, is_input=False):
    """ Encode data using provided encoder, for use in training/testing """
    return np.array(list(encoder.transform(
        text, reverse=is_input, fixed_length=S2S_PARAMS.message_len)),
                    dtype='int32')


encoder_for_lines = sklearn_encoder_for_data
encode_data = sklearn_encode_data
# encoder_for_lines = bpe_encoder_for_lines
# encode_data = bpe_encode_data

if __name__ == '__main__':
    train()
