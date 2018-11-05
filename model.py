import os
import pickle

import numpy as np

from keras import losses
from keras import utils
from keras import layers
from keras import models
from keras import optimizers
from keras import applications
from keras_contrib.layers.crf import CRF

from attention import Position_Embedding, Attention

MAX_PASSAGE_LENGTH = 21504
MAX_WORD_INDEX = 3326
units = 20
tag_num = 47


def model():
    passage_input = layers.Input(shape=(units,), dtype='int16')
    passage = layers.Embedding(MAX_WORD_INDEX + 1,
                               100,
                               # weights=[embedding_matrix],
                               input_length=units,
                               mask_zero=True)(passage_input)
    # passage = Position_Embedding()(passage)
    # p_encoder = layers.Conv1D(32, 7, activation='relu', padding='same')(passage)
    # p_encoder = layers.MaxPooling1D()(p_encoder)
    # p_encoder = layers.Conv1D(64, 7, activation='relu', padding='same')(p_encoder)
    # p_encoder = layers.MaxPooling1D()(p_encoder)
    # p_encoder = layers.Conv1D(128, 7, activation='relu', padding='same')(p_encoder)
    # p_encoder = layers.MaxPooling1D()(p_encoder)
    # p_encoder = layers.Conv1D(256, 7, activation='relu', padding='same')(p_encoder)
    # p_encoder = layers.MaxPooling1D()(p_encoder)
    # p_encoder = layers.Conv1D(512, 7, activation='relu', padding='same')(p_encoder)
    # p_encoder = layers.MaxPooling1D()(p_encoder)
    # p_encoder = CRF(46, sparse_target=True)(p_encoder)
    p_encoder = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(passage)
    # p_encoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(p_encoder)
    # p_encoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(p_encoder)
    # p_encoder = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(p_encoder)
    # p_encoder = passage
    # p_encoder = Attention(16, 32)([p_encoder, p_encoder, p_encoder])
    crf = CRF(tag_num, sparse_target=True)
    p_encoder = crf(p_encoder)

    # a_decoder = Attention(1, 4)([p_encoder, q_encoder, alt_encoder])
    # a_decoder = layers.Flatten()(a_decoder)
    # alternatives_input = layers.Flatten()(alternatives_input)
    # a_decoder = layers.Concatenate()([a_decoder, alternatives_input])
    # a_decoder = layers.GlobalMaxPooling1D()(a_decoder)

    output = p_encoder

    rc_model = models.Model(inputs=passage_input, outputs=output)
    opti = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    rc_model.compile(optimizer=opti, loss=crf.loss_function, metrics=[crf.accuracy])

    rc_model.summary()
    return rc_model


if __name__ == '__main__':
    model()
