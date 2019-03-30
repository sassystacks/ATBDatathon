import tensorflow as tf
import numpy as np
from model import Predictor
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

n_instances = x_train.shape[0]
res_x = x_train.shape[1]
res_y = x_train.shape[2]
batch_size = 256
n_epochs = 100


x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

model = Predictor(H=res_x,
                  W=res_y,
                  C=1,
                  l_rate=0.1,
                  n_filters=16,
                  n_layers=3,
                  n_outs=10,
                  p_type='classifier')

sess = tf.Session()
sess.run(tf.initializers.global_variables())

for n in range(n_epochs):
    for b in (range(0, n_instances, batch_size)):
        x = np.asarray(x_train[b:b+batch_size])
        t = np.asarray(t_train[b:b+batch_size])

        feed = {model.x: x,
                model.t: t,
                model.train: True}
        _, loss, y = sess.run([model.optimizer, model.loss, model.y], feed)
        print(t[0], np.argmax(y[0]), loss)
        plt.imshow(x[0, :, :, 0])
        plt.title(['Prediction: ', str(np.argmax(y[0]))])
        plt.show()
