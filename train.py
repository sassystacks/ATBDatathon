import tensorflow as tf
import numpy as np
from model import Predictor
import matplotlib.pyplot as plt


#mnist = tf.keras.datasets.mnist
#(x_train, t_train), (x_test, t_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0


dataset = np.load('dataset.npy')

dataset = np.asarray([d for d in dataset])

data_len = len(dataset)

split = int(data_len*0.7)

x_train = np.asarray([dataset[:split, 0][n] for n in range(split)])
t_train = np.asarray([dataset[:split, 1][n] for n in range(split)])

t_train = -(t_train-np.min(t_train))/(np.min(t_train - np.max(t_train)))

n_instances = x_train.shape[0]
res_x = x_train[0].shape[1]
res_y = x_train[0].shape[2]
num_c = x_train[0].shape[3]
batch_size = 64
n_epochs = 100


x_train = np.expand_dims(x_train, -1)
x_train = np.squeeze(np.swapaxes(x_train, 1, -1))
x_train = np.reshape(x_train, [len(x_train), res_x, res_y, -1])


model = Predictor(H=res_x,
                  W=res_y,
                  C=54,
                  l_rate=0.1,
                  n_filters=16,
                  n_layers=3,
                  n_outs=1,
                  p_type='regressor')

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
        print(t[0], (y[0]), loss)
