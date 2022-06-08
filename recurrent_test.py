import numpy as np
import tensorflow as tf
import keras
import cv2
from keras import layers, activations


def module(x, initializer, alpha=0.001):
    n = x.shape[1]

    y = x

    x = layers.Dense(n, use_bias=False,
        kernel_initializer=initializer)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = activations.relu(x, alpha=alpha)

    x = layers.Dense(n, use_bias=True,
        bias_initializer=initializer,
        kernel_initializer=initializer)(x)
    
    return layers.Add()([x, y])


def create_model(state_size):
    initializer = keras.initializers.RandomNormal(stddev=0.03)

    in_x = layers.Input(shape=(state_size))
    in_s = layers.Input(shape=(state_size))

    # reset gate
    r_x = module(in_x, initializer)
    r_s = module(in_s, initializer)
    # r_x = layers.Dense(state_size, use_bias=False)(in_x)
    # r_s = layers.Dense(state_size, use_bias=False)(in_s)
    r = layers.Add()([r_x, r_s])
    r = layers.Dense(state_size, use_bias=True,
        bias_initializer=keras.initializers.RandomUniform(-0.5, 1.5),
        kernel_initializer=initializer)(r)
    r = layers.Activation(activations.sigmoid)(r)

    # update gate
    u_x = module(in_x, initializer)
    u_s = module(in_s, initializer)
    # u_x = layers.Dense(state_size, use_bias=False)(in_x)
    # u_s = layers.Dense(state_size, use_bias=False)(in_s)
    u1 = layers.Add()([u_x, u_s])
    u1 = layers.Dense(state_size, use_bias=True,
        bias_initializer=keras.initializers.RandomUniform(-1.0, 4.0),
        kernel_initializer=initializer)(u1)
    u1 = layers.Activation(activations.sigmoid)(u1)
    u2 = layers.Lambda(lambda x: 1.0 - x)(u1)

    # candidate state
    s_x = module(in_x, initializer)
    s_s = module(layers.Multiply()([in_s, r]), initializer)
    # s_x = layers.Dense(state_size, use_bias=False)(in_x)
    # s_s = layers.Dense(state_size, use_bias=False)(in_s)
    s = layers.Add()([s_x, s_s])
    s = layers.Dense(state_size, use_bias=True,
        bias_initializer=initializer,
        kernel_initializer=initializer)(s)
    s = layers.Activation(activations.tanh)(s)

    # output state
    x = layers.Add()([ layers.Multiply()([in_s, u1]), layers.Multiply()([s, u2]) ])

    model = keras.Model(
        inputs = [in_x, in_s],
        outputs = x
    )

    model.summary()

    return model


def main():
    n_steps = 512
    state_size = 256

    model = create_model(state_size)

    inputs = []
    states = []

    state = tf.zeros((1, state_size))
    input = tf.random.uniform((1, state_size), -1.0, 1.0)
    for i in range(n_steps):
        state = model([input, state])
        input = tf.where(tf.random.uniform((1, state_size)) > 0.001,
            input, tf.random.uniform((1, state_size), -1.0, 1.0))

        inputs.append(input[0]*0.5 + 0.5)
        states.append(state.numpy()[0]*0.5 + 0.5)
    
    inputs_img = np.stack(inputs, axis=1)
    states_img = np.stack(states, axis=1)
    
    cv2.imshow("inputs", inputs_img)
    cv2.imshow("states", states_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()