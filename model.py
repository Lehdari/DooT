import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import activations
import random
from utils import *


class Model:
	def __init__(self):
		self.initializer = initializers.RandomNormal(stddev=0.04)
		self.state_size = 64
		self.state = np.zeros((self.state_size,))

		self.create_state_model(3)
		self.create_action_model()

		# combine the state and action models into one model and compile it
		action_outputs = self.action_model(self.state_outputs_state)
		self.combined_model = keras.Model(
			inputs=[self.state_inputs_image, self.state_inputs_state, self.state_inputs_action],
			outputs=action_outputs)
		self.combined_model.compile(
			loss="mean_squared_error",
			optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.05)
		)
	
	def module_dense(self, x, n, alpha=0.01, dropout=None):
		x = layers.Dense(n, kernel_initializer=self.initializer)(x)
		x = activations.relu(x, alpha=alpha)

		x = layers.BatchNormalization(axis=-1)(x)

		if dropout is not None:
			x = layers.Dropout(dropout)(x)
		
		return x
	
	def module_conv(self, x, n1, n2, k1=(3,3), k2=(3,3), dropout=None):
		x = layers.Conv2D(n1, k1, padding="same", kernel_initializer=self.initializer,
			strides=(2,2), activation="relu")(x)
		x = layers.Conv2D(n2, k2, padding="same", kernel_initializer=self.initializer,
			activation="relu")(x)

		x = layers.BatchNormalization(axis=-1)(x)

		if dropout is not None:
			x = layers.Dropout(dropout)(x)
		
		return x



	def create_state_model(self, n_channels):
		self.state_inputs_image = keras.Input(shape=(240, 320, n_channels))
		self.state_inputs_state = keras.Input(shape=(self.state_size))
		self.state_inputs_action = keras.Input(shape=(15))

		# image input branch
		x = layers.Conv2D(16, (3, 3), padding="same", kernel_initializer=self.initializer,
			activation="relu")(self.state_inputs_image)
		x = layers.BatchNormalization(axis=-1)(x)

		x = self.module_conv(x, 32, 32, dropout=0.4) #120x160
		x = self.module_conv(x, 64, 64, dropout=0.3) #60x80
		x = self.module_conv(x, 128, 128, dropout=0.2) #30x40

		x = layers.Conv2D(256, (3, 3), padding="same", kernel_initializer=self.initializer,
			strides=(2,2), activation="relu")(x) #15x20
		x = layers.Conv2D(256, (2, 3), kernel_initializer=self.initializer,
			activation="relu")(x) #14x18
		x = layers.BatchNormalization(axis=-1)(x)

		x = self.module_conv(x, 512, 512, k2=(1, 1)) #7x9

		x = layers.Conv2D(256, (1, 1), kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv2D(128, (1, 1), kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv2D(64, (1, 1), kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv2D(32, (1, 1), kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Flatten()(x)

		# concatenate with other inputs
		x = layers.concatenate([x, self.state_inputs_state, self.state_inputs_action])

		x = self.module_dense(x, 1024, dropout=0.5)
		x = self.module_dense(x, 512, dropout=0.3)
		x = self.module_dense(x, 256, dropout=0.1)
		x = self.module_dense(x, self.state_size*2)

		# linear activation in state output
		self.state_outputs_state =\
			layers.Dense(self.state_size,  kernel_initializer=self.initializer)(x)

		self.state_model = keras.Model(
			inputs=[self.state_inputs_image, self.state_inputs_state, self.state_inputs_action],
			outputs=self.state_outputs_state)
		self.state_model.summary()

	def create_action_model(self):
		self.action_inputs_state = keras.Input(shape=(self.state_size))

		x = self.module_dense(self.action_inputs_state, 512, dropout=0.5)
		x = self.module_dense(x, 256, dropout=0.3)
		x = self.module_dense(x, 128, dropout=0.1)
		x = self.module_dense(x, 64)

		self.action_outputs_action = layers.Dense(15, kernel_initializer=self.initializer,\
			activation="tanh")(x)

		self.action_model = keras.Model(inputs=self.action_inputs_state,
			outputs=self.action_outputs_action)
		self.action_model.summary()

	def advance(self, frame, action):
		# convert action into continuous domain so it can be passed to action model
		action = convert_action_to_continuous(action)
		
		self.state = self.state_model.predict([
			np.expand_dims(frame, 0),
			np.expand_dims(self.state, 0),
			np.expand_dims(action, 0)])[0]

	"""
	Reset state (after an episode)
	"""
	def reset_state(self):
		self.state = np.zeros((self.state_size,))

	"""
	Predict action from the state of the model
	return: list length of 15: 14 booleans and 1 float
	"""
	def predict_action(self):
		#print("state: {}".format(self.state)) # TODO REMOVE
		action = self.action_model.predict(np.expand_dims(self.state,0))[0]
		return convert_action_to_mixed(action)

	def train(self, frames_in, states_in, actions_in, actions_out):
		frames_in = np.asarray(frames_in)
		states_in = np.asarray(states_in)
		actions_in = np.asarray(actions_in)
		actions_out = np.asarray(actions_out)

		self.combined_model.fit(x=[frames_in, states_in, actions_in], y=actions_out,
			batch_size=32, epochs=8, shuffle=True)

	def save_model(self, state_model_filename, action_model_filename):
		self.state_model.save(state_model_filename)
		self.action_model.save(action_model_filename)
	
	def load_model(self, state_model_filename, action_model_filename):
		self.state_model = keras.models.load_model(state_model_filename)
		self.action_model = keras.models.load_model(action_model_filename)
		# combine the state and action models into one model and compile it
		action_outputs = self.action_model(self.state_outputs_state)
		self.combined_model = keras.Model(
			inputs=[self.state_inputs_image, self.state_inputs_state, self.state_inputs_action],
			outputs=action_outputs)
		self.combined_model.compile(
			loss="mean_squared_error",
			optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.05)
		)