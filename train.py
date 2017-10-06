
from bpe import Encoder
from gru_model import Seq2SeqConfig, build_model
import numpy as np

S2S_PARAMS = Seq2SeqConfig(
    message_len=30,
    batch_size=256,
    context_size=200,
    embed_size=100,
    use_cuda=True,
    vocab_size=2**13,
    start_token='<start/>',
)


def train():
    x_lines = open('data/x.txt').read().split('\n')
    y_lines = open('data/y.txt').read().split('\n')

    encoder = encoder_for_lines(S2S_PARAMS, x_lines + y_lines)

    model = build_model(S2S_PARAMS, encoder.word_vocab[S2S_PARAMS.start_token])

    x = encode_data(encoder, x_lines, is_input=True)
    y = encode_data(encoder, y_lines, is_input=False)

    for idx in range(1000):
        print("Training in epoch " + str(idx))
        #print(f"Training in epoch {idx+1}...")
        model.train_epoch(x, y)


def encoder_for_lines(cfg: Seq2SeqConfig, lines) -> Encoder:
    """ Calculate BPE encoder for provided lines of text """
    encoder = Encoder(vocab_size=cfg.vocab_size, required_tokens=[cfg.start_token])
    encoder.fit(lines)
    encoder.save('latest_encoder.json')
    return encoder


def encode_data(encoder, text, is_input=False):
    """ Encode data using provided encoder, for use in training/testing """
    return np.array(list(encoder.transform(
        text, reverse=is_input, fixed_length=S2S_PARAMS.message_len)),
                    dtype='int32')


if __name__ == '__main__':
    train()
