import os
import pickle

import numpy as np

from keras import utils
from keras import layers
from keras import models
from keras import optimizers
from keras import applications

from attention import Position_Embedding, Attention

MAX_ALT_NUM = 3
MAX_AN_LENGTH = 60
MAX_TRI_AN_LENGTH = 97
MAX_QUES_LENGTH = 30
MAX_PASSAGE_LENGTH = 300
MAX_WORD_INDEX = 13652  # 175749

def model():
    passage_input = layers.Input(shape=(MAX_PASSAGE_LENGTH,), dtype='int32')
    passage = layers.Embedding(MAX_WORD_INDEX + 1,
                               300,
                               weights=[embedding_matrix],
                               input_length=MAX_PASSAGE_LENGTH,
                               trainable=False)(passage_input)
    passage = Position_Embedding()(passage)
    question_input = layers.Input(shape=(MAX_QUES_LENGTH,))
    question = layers.Embedding(MAX_WORD_INDEX + 1,
                                300,
                                weights=[embedding_matrix],
                                input_length=MAX_PASSAGE_LENGTH,
                                trainable=False)(question_input)
    question = Position_Embedding()(question)
    alternatives_input = layers.Input(shape=(MAX_TRI_AN_LENGTH * 3,))
    alternatives = layers.Embedding(MAX_WORD_INDEX + 1,
                                    300,
                                    weights=[embedding_matrix],
                                    input_length=MAX_PASSAGE_LENGTH,
                                    trainable=False)(alternatives_input)

    # alternatives = layers.GlobalMaxPooling2D()(alternatives)
    alt_encoder = layers.Bidirectional(layers.LSTM(2, return_sequences=True))(alternatives)
    p_encoder = layers.Bidirectional(layers.LSTM(2, return_sequences=True))(passage)
    q_encoder = layers.Bidirectional(layers.LSTM(2, return_sequences=True))(question)

    alt_encoder = layers.MaxPooling1D()(alt_encoder)
    alt_encoder = layers.Flatten()(alt_encoder)
    p_encoder = layers.MaxPooling1D()(p_encoder)
    p_encoder = layers.Flatten()(p_encoder)
    q_encoder = layers.MaxPooling1D()(q_encoder)
    q_encoder = layers.Flatten()(q_encoder)

    a_decoder = layers.Concatenate()([p_encoder, q_encoder, alt_encoder])
    # a_decoder = Attention(1, 4)([p_encoder, q_encoder, alt_encoder])
    # a_decoder = layers.Flatten()(a_decoder)
    # alternatives_input = layers.Flatten()(alternatives_input)
    # a_decoder = layers.Concatenate()([a_decoder, alternatives_input])
    # a_decoder = layers.GlobalMaxPooling1D()(a_decoder)

    output_alt = layers.Dense(MAX_TRI_AN_LENGTH * 3)(a_decoder)
    output = layers.Dense(60)(a_decoder)

    rc_model = models.Model(inputs=[passage_input, question_input, alternatives_input], output=[output_alt, output])
    opti = optimizers.Adam(lr=1e-1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    rc_model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=["accuracy"])

    rc_model.summary()
    return rc_model
