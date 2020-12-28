import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
import random


class Model:
	def __init__(self):
		self.initializer = initializers.RandomNormal(stddev=0.04)
		self.state_size = 64
		#self.state = np.random.rand(self.state_size)
		self.state = np.zeros((self.state_size,))

		self.create_state_model(3)
		self.create_action_model()

	def create_state_model(self, n_channels):
		inputs_image = keras.Input(shape=(240, 320, n_channels))
		inputs_state = keras.Input(shape=(self.state_size))
		inputs_action = keras.Input(shape=(15))
		x = layers.Conv2D(16, (3, 3), padding="same", kernel_initializer=self.initializer,
			activation="relu")(inputs_image)
		x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer=self.initializer,
			strides=(2,2), activation="relu")(x) #120x160
		x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer=self.initializer,
			strides=(2,2), activation="relu")(x) #60x80
		x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer=self.initializer,
			strides=(2,2), activation="relu")(x) #30x40
		x = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv2D(256, (3, 3), padding="same", kernel_initializer=self.initializer,
			strides=(2,2), activation="relu")(x) #15x20
		x = layers.Conv2D(256, (2, 3), kernel_initializer=self.initializer,
			activation="relu")(x) #14x18
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Conv2D(512, (3, 3), padding="same", kernel_initializer=self.initializer,
			strides=(2,2), activation="relu")(x) #7x9
		x = layers.Conv2D(512, (1, 1), kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
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
		x = layers.concatenate([x, inputs_state, inputs_action])
		x = layers.Dense(512,  kernel_initializer=self.initializer, activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Dense(512,  kernel_initializer=self.initializer, activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Dense(256,  kernel_initializer=self.initializer, activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Dense(self.state_size*2,  kernel_initializer=self.initializer,
			activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		# linear activation in state output
		outputs_state = layers.Dense(self.state_size,  kernel_initializer=self.initializer)(x)

		self.state_model = keras.Model(
			inputs=[inputs_image, inputs_state, inputs_action],
			outputs=outputs_state)
		self.state_model.summary()

		self.state_model.compile(
			loss="mean_squared_error",
			optimizer=keras.optimizers.Adam()
		)

	def create_action_model(self):
		inputs_state = keras.Input(shape=(self.state_size))
		x = layers.Dense(512, kernel_initializer=self.initializer, activation="relu")(inputs_state)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Dense(256, kernel_initializer=self.initializer, activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Dense(128, kernel_initializer=self.initializer, activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		outputs_action = layers.Dense(15, kernel_initializer=self.initializer, activation="tanh")(x)

		self.action_model = keras.Model(inputs=inputs_state, outputs=outputs_action)
		self.action_model.summary()

		self.action_model.compile(
			loss="mean_squared_error",
			optimizer=keras.optimizers.Adam()
		)

	def advance(self, frame, action):
		# convert action into continuous domain so it can be passed to action model
		action_cont = np.where(action, 1.0, -1.0)
		action_cont[14] = action[14]

		# TODO TEMP
		#frame = np.zeros_like(frame)
		#self.state = np.zeros_like(self.state)
		#action_cont = np.zeros_like(action_cont)
		
		self.state = self.state_model.predict([
			np.expand_dims(frame, 0),
			np.expand_dims(self.state, 0),
			np.expand_dims(action_cont, 0)])[0]


	"""
	Predict action from the state of the model
	return: list length of 15: 14 booleans and 1 float
	"""
	def predict_action(self):
		#print("state: {}".format(self.state)) # TODO REMOVE
		prediction = self.action_model.predict(np.expand_dims(self.state,0))[0]
		action = np.where(prediction > 0.0, True, False).tolist()
		action[14] = prediction[14]*100.0
		return action

	"""
	return: list length of 15: 14 booleans and 1 float
	"""
	def get_random_action(self):
		random_action = random.choices([True, False], k=14)
		random_action.append(random.gauss(0, 25.0))
		return random_action

	def save_model(self, filename):
		self.model.save(filename)