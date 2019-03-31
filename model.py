import tensorflow as tf


class Predictor():
    def __init__(self,
                 H=32,
                 W=32,
                 C=1,
                 l_rate=0.01,
                 n_filters=32,
                 n_layers=5,
                 n_outs=3,
                 p_type='classifier'):

        self._init_network(H, W, C, n_filters, n_layers, n_outs)
        self.p_type = p_type
        self.loss = self._init_loss()
        self.optimizer = self._init_optimizer(l_rate)

    def _init_network(self, H, W, C, filters, n_layers, n_outs):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, shape=(None, H, W, C))
            self.t = tf.placeholder(tf.float32, shape=(None))
            self.train = tf.placeholder(tf.bool)

        with tf.name_scope('Convolutional_Layers'):
            self.conv = []

            self.conv.append(
                tf.contrib.layers.batch_norm(self.x,
                                             center=True,
                                             scale=True,
                                             is_training=1))

            for n in range(n_layers):
                self.conv.append(tf.layers.conv2d(self.conv[-1],
                                                  filters**n,
                                                  [3, 3],
                                                  strides=2,
                                                  padding='same',
                                                  activation=tf.nn.relu))
                self.conv.append(
                    tf.contrib.layers.batch_norm(self.conv[-1],
                                                 center=True,
                                                 scale=True,
                                                 is_training=1))

            flat = tf.layers.flatten(self.conv[-1])

            self.fc = []
            self.fc.append(tf.layers.dense(flat, 32, activation=tf.nn.relu))
            self.fc.append(
                tf.contrib.layers.batch_norm(self.fc[-1],
                                             center=True,
                                             scale=True,
                                             is_training=1))

            self.y = tf.layers.dense(self.fc[-1], n_outs, activation=None)

    def _init_loss(self):

        if self.p_type is 'classifier':
            return tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.t,
                    logits=self.y))

        elif self.p_type is 'regressor':
            return tf.reduce_mean(tf.square(self.t-self.y))

    def _init_optimizer(self, l_rate):
        return tf.train.AdamOptimizer(l_rate).minimize(self.loss)
