__author__ = 'fuadissa'

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow import nn


def build_graph(training_setting):
    tf.reset_default_graph()
    graph = tf.Graph()

    with graph.as_default():
        with tf.name_scope('inputs') as name_scope:
            sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
            X_sent = tf.placeholder(tf.int32, [None, training_setting['maximum_sent_length']], name='x_sent')
            y_slot = tf.placeholder(tf.int32, [None, training_setting['maximum_sent_length']], name='y_slot')
            y_intent = tf.placeholder(tf.int32, [None], name='y_intent')
            y_intent = tf.one_hot(y_intent, depth=training_setting['intent_num'], dtype=tf.int32)
            dropout = tf.placeholder(tf.float32, shape=[], name='dropout')

            pretrained_embeddings_input = tf.placeholder(tf.float32, shape=[training_setting['pretrained_vocab_length'],
                                                                            training_setting['embedding_size']],
                                                         name='pretrained_embeddings_ph')

        with tf.name_scope('embedding') as name_scope:
            reserved_embeddings = tf.Variable(tf.random_uniform([training_setting['reserved_vocab_length'],
                                                                 training_setting['embedding_size']], -1.0, 1.0),
                                              trainable=True, name='reserved_embeddings')

            if training_setting['use_pretrained_embeddings']:
                pretrained_embeddings = tf.Variable(tf.random_uniform([training_setting['pretrained_vocab_length'],
                                                                       training_setting['embedding_size']], -1.0, 1.0),
                                                    trainable=False, name='pretrained_embeddings')

                assign_pretrained_embeddings = tf.assign(pretrained_embeddings, pretrained_embeddings_input,
                                                         name='assign_pretrained_embeddings')
            else:

                pretrained_embeddings = tf.Variable(tf.random_uniform([training_setting['pretrained_vocab_length'],
                                                                       training_setting['embedding_size']], -1.0, 1.0),
                                                    trainable=True, name='pretrained_embeddings')

            X_sent = tf.where(tf.less(X_sent, tf.constant(training_setting['reserved_vocab_length'])), X_sent * 2,
                              (X_sent - training_setting['reserved_vocab_length']) * 2 + 1)

            word_embeddings_sent = tf.nn.embedding_lookup([reserved_embeddings, pretrained_embeddings], X_sent,
                                                          name='word_embeddings_sent')

            # word_embeddings_sent = tf_print(word_embeddings_sent, 'word_embeddings_sent')

        with tf.name_scope('gru_cell_sent') as name_scope:
            gru_forward_sent = rnn.DropoutWrapper(rnn.GRUCell(training_setting['hidden_units']),
                                                  output_keep_prob=dropout)
            gru_backward_sent = rnn.DropoutWrapper(rnn.GRUCell(training_setting['hidden_units']),
                                                   output_keep_prob=dropout)

            (gru_output_forward, gru_output_backward), _ = nn.bidirectional_dynamic_rnn(gru_forward_sent,
                                                                                        gru_backward_sent,
                                                                                        word_embeddings_sent,
                                                                                        dtype=tf.float32,
                                                                                        sequence_length=sequence_length,
                                                                                        scope=name_scope)

            bidirectional_gru_output_sent = tf.concat(axis=2, values=(gru_output_forward, gru_output_backward),
                                                      name='output_sent')

            # bidirectional_gru_output_sent = tf_print(bidirectional_gru_output_sent, 'bidirectional_gru_output_sent')

        with tf.name_scope('pooling') as name_scope:
            W = tf.Variable(tf.random_normal([2 * training_setting['hidden_units']], name='attention_weight'))
            b = tf.Variable(tf.random_normal([1]), name='attention_bias')
            # W = tf_print(W, 'W')
            attentions = tf.reduce_sum(tf.multiply(W, bidirectional_gru_output_sent), axis=2) + b
            attentions = tf.nn.softmax(attentions)
            # attentions = tf_print(attentions, 'attentions')

            expand_attentions = tf.expand_dims(attentions, 1)
            transpose_outputs = tf.transpose(bidirectional_gru_output_sent, perm=[0, 2, 1])

            attentions_output = tf.reduce_sum(tf.transpose(tf.multiply(expand_attentions, transpose_outputs),
                                                           perm=[0, 2, 1]), axis=1)
            # attentions_output = tf_print(attentions_output, 'attentions_output')

        with tf.name_scope('slot') as name_scope:
            # attentions_output = tf_print(attentions_output, 'attention output')
            W_projection_slot = tf.get_variable("W_projection_slot",
                                                shape=[64, training_setting['slot_num']])
            b_projection_slot = tf.get_variable("b_projection_slot", shape=[training_setting['slot_num']])
            logits = []
            hidden_states_list = []
            for i in range(training_setting['maximum_sent_length']):
                feature = bidirectional_gru_output_sent[:, i, :]
                hidden_states = tf.layers.dense(feature, 64, activation=tf.nn.tanh)
                output = tf.matmul(hidden_states, W_projection_slot) + b_projection_slot
                logits.append(output)
                hidden_states_list.append(hidden_states)

            logits_slots = tf.stack(logits, axis=1)
            y_pred_slot = tf.argmax(logits_slots, axis=2, name="y_pred")

        with tf.name_scope('intent') as name_scope:
            W_mlp = tf.Variable(
                tf.random_normal([2 * training_setting['hidden_units'], training_setting['intent_num']]))
            b_mlp = tf.Variable(tf.random_normal([training_setting['intent_num']]))

            logits_intent = tf.matmul(attentions_output, W_mlp) + b_mlp
            y_pred_intent = tf.argmax(logits_intent, 1, name='y_pred')

        with tf.name_scope('loss') as name_scope:
            mask = tf.to_float(tf.not_equal(sequence_length, 0))

            loss_slot = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_slot, logits=(logits_slots))
            loss_slot = tf.reduce_sum(loss_slot, axis=1) / training_setting['maximum_sent_length']
            loss_slot = tf.reduce_mean(loss_slot)

            loss_intent = tf.nn.softmax_cross_entropy_with_logits(labels=y_intent, logits=logits_intent)
            loss_intent = tf.reduce_mean(loss_intent)

            # l2_losses = tf.add_n(
            #     [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
            # weights_intent = tf.nn.sigmoid(tf.cast(self.global_step / 1000, dtype=tf.float32)) / 2

            loss = loss_slot + loss_intent
            loss = tf.identity(loss, name='loss')

            # loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_intent, name='loss')

        with tf.name_scope('optimizer') as name_scope:
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=training_setting['learning_rate']).minimize(loss,
                                                                                                             name='optimizer')
    return graph


def tf_print(var, message):
    return tf.Print(var, [tf.shape(var), var], message=message)
