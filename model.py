import os
import pickle

import numpy as np

from keras import utils
from keras import layers
from keras import models
from keras import optimizers
from keras import applications

from attention import Position_Embedding, Attention

MAX_PASSAGE_LENGTH = 21504
MAX_WORD_INDEX = 3244


def model():
    passage_input = layers.Input(shape=(MAX_PASSAGE_LENGTH,), dtype='int32')
    passage = layers.Embedding(MAX_WORD_INDEX + 1,
                               300,
                               # weights=[embedding_matrix],
                               input_length=MAX_PASSAGE_LENGTH,
                               trainable=False)(passage_input)
    passage = Position_Embedding()(passage)
    p_encoder = layers.Bidirectional(layers.LSTM(2, return_sequences=True))(passage)
    p_encoder = layers.Bidirectional(layers.LSTM(2, return_sequences=True))(p_encoder)

    p_encoder = layers.MaxPooling1D()(p_encoder)
    p_encoder = layers.Flatten()(p_encoder)

    # a_decoder = Attention(1, 4)([p_encoder, q_encoder, alt_encoder])
    # a_decoder = layers.Flatten()(a_decoder)
    # alternatives_input = layers.Flatten()(alternatives_input)
    # a_decoder = layers.Concatenate()([a_decoder, alternatives_input])
    # a_decoder = layers.GlobalMaxPooling1D()(a_decoder)

    output = layers.Dense(21504, activation='relu')(p_encoder)

    rc_model = models.Model(inputs=passage_input, outputs=output)
    opti = optimizers.Adam(lr=1e-1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    rc_model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=["accuracy"])

    rc_model.summary()
    return rc_model


if __name__ == '__main__':
    model()
