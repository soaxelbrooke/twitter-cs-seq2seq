
import os
import re
import random

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
    message_len=25,
    batch_size=int(os.environ.get('BATCH_SIZE', '2')),
    context_size=100,
    embed_size=200,
    use_cuda=True,
    vocab_size=2**13,
    start_token='<start/>',
    encoder_layers=int(os.environ.get('ENCODER_LAYERS', '2')),
    learning_rate=float(os.environ.get('LEARNING_RATE', '0.005')),
    teacher_force_ratio=float(os.environ.get('TEACHER_FORCE', '0.5')),
)

AT_TOKEN = '<at/>'
HASH_TOKEN = '<hashtag/>'
SIGNATURE_TOKEN = '<signature/>'
PAD_TOKEN = '_'

LIMIT = int(os.environ.get('LIMIT', '10000000'))


def train():
    x_lines = [*toolz.take(LIMIT, open('data/x.txt').read().lower().split('\n'))]
    y_lines = [*toolz.take(LIMIT, open('data/y.txt').read().lower().split('\n'))]

    encoder = encoder_for_lines(S2S_PARAMS, x_lines + y_lines)

    try:
        start_idx = encoder.word_vocab[S2S_PARAMS.start_token]
        pad_idx = encoder.word_vocab[PAD_TOKEN]
    except AttributeError:
        start_idx = int(encoder.vocabulary_[S2S_PARAMS.start_token])
        pad_idx = encoder.vocabulary_[PAD_TOKEN]

    reverse_enc = {idx: word for word, idx in encoder.vocabulary_.items()}
    model = build_model(S2S_PARAMS, start_idx, pad_idx)

    x = encode_data(encoder, x_lines, is_input=True)
    y = encode_data(encoder, y_lines, is_input=False)

    print(x.shape, y.shape)

    x = x[:S2S_PARAMS.batch_size*int(len(x)/S2S_PARAMS.batch_size)]
    y = y[:S2S_PARAMS.batch_size*int(len(y)/S2S_PARAMS.batch_size)]

    test_x = x[:S2S_PARAMS.batch_size]
    losses = []

    if USE_COMET:
        experiment = Experiment(api_key="DQqhNiimkjP0gK6c8iGz9orzL", log_code=True)
        experiment.log_multiple_params(S2S_PARAMS._asdict())
        for idx in range(1000):
            print("Shuffling data...")
            random_idx = random.sample([*range(len(x))], len(x))
            x = x[random_idx]
            y = y[random_idx]
            print("Training in epoch " + str(idx))
            losses.append(model.train_epoch(x, y, experiment=experiment))
            experiment.log_epoch_end(idx)
            print('Loss history: {}'.format(', '.join(['{:.4f}'.format(loss) for loss in losses])))
            test_y = model.predict(test_x)
            for i in range(min([3, S2S_PARAMS.batch_size])):
                print('> ' + ' '.join(reverse_enc.get(idx, '<unk/>') for idx in list(test_y[i])))
    else:
        for idx in range(1000):
            print("Training in epoch " + str(idx))
            model.train_epoch(x, y)


def sklearn_encoder_for_data(cfg: Seq2SeqConfig, lines):
    logging.info("Fitting sklearn count vectorizer...")
    cv = CountVectorizer(tokenizer=casual_tokenize, max_features=cfg.vocab_size - 2)
    signature_regex = re.compile(r'[\*^/][a-z][a-z][a-z]?(?![\*])')
    fixed_lines = [signature_regex.sub(SIGNATURE_TOKEN, text) for text in lines]
    cv.fit(fixed_lines + [cfg.start_token] * 10000 + [AT_TOKEN] * 10000 + [HASH_TOKEN] * 10000
           + [SIGNATURE_TOKEN] * 10000 + [PAD_TOKEN] * 10000)
    return cv


def bpe_encoder_for_lines(cfg: Seq2SeqConfig, lines) -> Encoder:
    """ Calculate BPE encoder for provided lines of text """
    encoder = Encoder(vocab_size=cfg.vocab_size,
                      required_tokens=[cfg.start_token, AT_TOKEN, HASH_TOKEN, SIGNATURE_TOKEN])
    encoder.fit(lines)
    encoder.save('latest_encoder.json')
    return encoder


def sklearn_encode_data(encoder, text, is_input=False):
    encoded = np.ones((len(text), S2S_PARAMS.message_len),
                      dtype='int32') * encoder.vocabulary_[PAD_TOKEN]
    unk = encoder.max_features

    unk_count = 0
    at_count = 0
    hash_count = 0
    signature_count = 0
    known_count = 0

    signature_regex = re.compile(r'[\*^/][a-z][a-z][a-z]?(?![\*])')

    for row_idx, line in enumerate(text):
        for col_idx, token in enumerate(toolz.take(S2S_PARAMS.message_len, encoder.tokenizer(line))):
            if token.startswith('@') and token not in encoder.vocabulary_:
                at_count += 1
                word_idx = encoder.vocabulary_[AT_TOKEN]
            elif token.startswith('#') and token not in encoder.vocabulary_:
                hash_count += 1
                word_idx = encoder.vocabulary_[HASH_TOKEN]
            elif token not in encoder.vocabulary_ and signature_regex.match(token):
                signature_count += 1
                word_idx = encoder.vocabulary_[SIGNATURE_TOKEN]
            elif token in encoder.vocabulary_:
                known_count += 1
                word_idx = encoder.vocabulary_[token]
            else:
                unk_count += 1
                word_idx = unk
            _col_idx = S2S_PARAMS.message_len - col_idx - 1 if is_input else col_idx
            encoded[row_idx][_col_idx] = word_idx

    logging.warning(f"{100 * unk_count / (signature_count + at_count + hash_count + known_count)}% "
                    f"OOV")
    logging.warning(f"Found {unk_count} unks, {at_count} unknown @s, "
                    f"{hash_count} unknown hashtags, {signature_count} unknown signatures, "
                    f"{known_count} known words.")
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
