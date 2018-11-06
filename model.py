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

from keras_self_attention import SeqSelfAttention
from keras_pos_embd import PositionEmbedding
from keras_multi_head import MultiHead

MAX_PASSAGE_LENGTH = 21504
MAX_WORD_INDEX = 3327
units = 50
tag_num = 47


def multi_head(n_head, at_units, width, pre_layer, activation='sigmoid'):
    layers_list = []
    for i in range(n_head):
        self_at = SeqSelfAttention(units=at_units, attention_width=width, attention_activation=activation)(pre_layer)
        layers_list.append(self_at)
    mt = layers.Concatenate()(layers_list)
    return mt


def model():
    passage_input = layers.Input(shape=(units,), dtype='int16')
    passage_embd = layers.Embedding(MAX_WORD_INDEX + 1,
                                    100,
                                    # weights=[embedding_matrix],
                                    input_length=units,
                                    mask_zero=True)(passage_input)
    # passage_posi = PositionEmbedding(input_dim=MAX_WORD_INDEX + 1,  # The maximum absolute value of positions.
    #                                  output_dim=100,  # The dimension of embeddings.
    #                                  mask_zero=False,
    #                                  # The index that presents padding (because `0` will be used in relative positioning).
    #                                  input_shape=(None,),
    #                                  name='Pos-Embd', )(passage_input)
    # passage = layers.Add()([passage_embd, passage_posi])
    passage = passage_embd
    p_encoder = layers.Bidirectional(layers.LSTM(int(tag_num / 2), return_sequences=True))(passage)
    p_encoder = layers.Bidirectional(layers.LSTM(int(tag_num / 2), return_sequences=True))(p_encoder)

    p_encoder = layers.LSTM(tag_num, return_sequences=True)(p_encoder)
    p_encoder = layers.LSTM(tag_num, return_sequences=True)(p_encoder)
    # p_encoder = passage
    # p_encoder = SeqSelfAttention(attention_activation='sigmoid')(p_encoder)
    # p_encoder = multi_head(2, 1000, tag_num, p_encoder)

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
