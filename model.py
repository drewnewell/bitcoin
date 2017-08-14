from __future__ import print_function

import pickle
import tensorflow as tf
import numpy as np
import binascii


class NonceModel:

    def __init__(
            self,
            batch_size = 128,
            layers = 5,
            restore_model = True,
            data_file = 'data/nonce.pkl',
            checkpoint = 'model/model.ckpt'):

        self.checkpoint = checkpoint
        self.data_file = data_file
        self.batch_size = batch_size

        self.build_model(layers)
        self.sess = tf.Session()

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        if restore_model:
            self.saver.restore(self.sess, self.checkpoint)

    def build_model(self, layers=5):

        self.inputs = tf.placeholder(tf.float32, [None, 608])
        self.labels = tf.placeholder(tf.float32, [None, 32])
        self.learning_rate = tf.placeholder(dtype=tf.float32)

        def fc_layer(inputs, units=512):
            fc_out = tf.layers.dense(inputs, units)
            return tf.contrib.layers.batch_norm(fc_out, activation_fn=tf.nn.relu)

        fc_output = self.inputs
        for i in range(layers):
            fc_output = fc_layer(fc_output)

        self.output = tf.layers.dense(fc_output, 32)

        self.loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels = self.labels,
            logits = self.output
        )

        self.optimize = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def get_batches(self, batch_size, source='train'):
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)

        val_split = int(len(data) * .3)

        data = {
            'train': data[:-val_split],
            'val': data[-val_split:]
        }

        num_batches = len(data[source]) // batch_size

        for i in range(num_batches):
            header_size = 608 # length in bits, minus nonce
            nonce_size = 32 # length in bits

            heights, header_bytes, nonce_bytes = zip(*data[source][i * batch_size : (i + 1) * batch_size])

            batch_heights = np.array(heights, dtype=np.int32)
            batch_headers = np.zeros((batch_size, header_size), dtype=np.int32)
            batch_nonces = np.zeros((batch_size, nonce_size), dtype=np.int32)

            for j in range(batch_size):
                header_binstring = ''.join(['{:08b}'.format(int(binascii.hexlify(a), 16)) for a in header_bytes[j]])
                batch_headers[j] = np.array([int(s) for s in header_binstring])

                nonce_int = int(binascii.hexlify(nonce_bytes[j]), 16)
                change_bits = np.random.randint(0, 31, 12)
                change_filter = 0
                for cb in change_bits:
                    change_filter = change_filter | cb
                nonce_int = nonce_int ^ change_filter

                nonce_binstring = '{:032b}'.format(nonce_int)
                batch_nonces[j] = np.array([int(s) for s in nonce_binstring])

            yield batch_heights, batch_headers, batch_nonces

    def train(self, epochs=10, learning_rate=0.001):

        for epoch_i in range(epochs):
            avg_train_loss = 0.0
            for batch_i, (heights, headers, nonces) in enumerate(self.get_batches(self.batch_size)):
                _, batch_loss = self.sess.run(
                    [self.optimize, self.loss],
                    feed_dict = {
                        self.inputs: headers,
                        self.labels: nonces,
                        self.learning_rate: learning_rate
                    })

                avg_train_loss += batch_loss

                print('train: epoch {}, batch {}, error {:0.4f}.....\r'.format(
                    epoch_i, batch_i, batch_loss), end='')

            print('train: epoch {}, loss: {:.04f}.............'.format(
                epoch_i, avg_train_loss / batch_i))

            avg_val_loss = 0.0

            # validation check
            for batch_i, (heights, headers, nonces) in enumerate(self.get_batches(self.batch_size, source='val')):
                batch_loss = self.sess.run(
                    self.loss,
                    feed_dict = {
                        self.inputs: headers,
                        self.labels: nonces
                    })

                avg_val_loss += batch_loss

                print('validation: epoch {}, batch {}, error {:0.4f}.....\r'.format(
                    epoch_i, batch_i, batch_loss), end='')

            print('validation: epoch {}, loss: {:.04f}...........'.format(
                epoch_i, avg_val_loss / batch_i))

