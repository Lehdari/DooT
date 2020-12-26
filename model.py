import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

from collections import deque
import random

class Model:
	def __init__(self, reward):
		self.initializer = initializers.RandomNormal(stddev=0.04)
		self.create_model(1)

		self.reward = reward
		self.memory = deque(maxlen=2000)
		self.gamma = 0.85
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.005
		#self.tau = .125 Double DQN param, used later

	def create_model(self, n_channels):
		inputs = keras.Input(shape=(240, 320, n_channels))
		x = layers.Conv2D(16, (3, 3), padding="same", kernel_initializer=self.initializer,
			activation="relu")(inputs)
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
		x = layers.Flatten()(x)
		x = layers.Dense(256, activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Dense(256, activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		x = layers.Dense(128, activation="relu")(x)
		x = layers.BatchNormalization(axis=-1)(x)
		outputs = layers.Dense(15, activation="tanh")(x)

		self.model = keras.Model(inputs=inputs, outputs=outputs)
		self.model.summary()

		self.model.compile(
			loss="mean_squared_error",
			optimizer=keras.optimizers.Adam()
		)

	def train_batch(self, x, y):
		self.model.train_on_batch(x, y)

	"""
	return: list length of 15: 14 booleans and 1 float
	"""
	def predict_action(self, x):
		prediction = self.model.predict(x)[0]
		action = np.where(prediction > 0.0, True, False).tolist()
		action[14] = prediction[14]*100.0
		return action

	"""
	return: list length of 15: 14 booleans and 1 float
	"""
	def get_random_action(self):
		random_action = random.choices([True, False], k=14)
		random_action.append(random.uniform(-100.0, 100.0))
		return random_action

	"""
	Perform one step;
	- create action
	- get reward
	- update game state (happens in make_action under the hood)

	return: reward (1D float)
	"""
	def step(self, game):
		state = game.get_state()

		#TODO: stack frames
		screen_buf = state.screen_buffer

		"""
		Epsilon-greedy algorithm
		With probability epsilon choose a random action ("explore")
		With probability 1-epsilon choose best known action ("exploit")
		"""

		action = self.predict_action(np.expand_dims(screen_buf,0))

		self.epsilon *= self.epsilon_decay
		self.epsilon = max(self.epsilon_min, self.epsilon)
		if np.random.random() < self.epsilon:
			action = self.get_random_action()

		# Intentionally ignore the reward the game gives
		game.make_action(action)

		# Instead, use our own reward system
		reward = self.reward.get_reward(game)
		return reward
		