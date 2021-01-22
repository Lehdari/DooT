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
		self.initializer = initializers.RandomNormal(stddev=0.07)
		self.state_size = 1024
		self.image_enc_size = 1024

		self.state = np.zeros((self.state_size,))
		self.action_predict_step_size = 0.01

		self.create_image_model(3)
		self.create_state_model()
		self.create_action_model()
		self.create_forward_model()
		self.create_inverse_model()
		self.create_reward_model()

		self.combine_model()
	
	def module_dense(self, x, n, alpha=0.01, dropout=None):
		x = layers.Dense(n, kernel_initializer=self.initializer)(x)
		x = activations.relu(x, alpha=alpha)

		x = layers.BatchNormalization(axis=-1)(x)

		if dropout is not None:
			x = layers.Dropout(dropout)(x)
		
		return x
	
	def module_conv(self, x, n1, n2, k1=(3,3), k2=(3,3), s1=(2,2), s2=(1,1),
		p1="same", p2="same", dropout=None):

		x = layers.Conv2D(n1, k1, padding=p1, kernel_initializer=self.initializer,
			strides=s1, activation="relu")(x)
		x = layers.Conv2D(n2, k2, padding=p2, kernel_initializer=self.initializer,
			strides=s2, activation="relu")(x)

		x = layers.BatchNormalization(axis=-1)(x)

		if dropout is not None:
			x = layers.Dropout(dropout)(x)
		
		return x
	
	def create_image_model(self, n_channels):
		self.model_image_i_image = keras.Input(shape=(240, 320, n_channels))

		x = self.module_conv(self.model_image_i_image, 32, 64,
			k1=(3,2), s1=(3,2), k2=(3,3), s2=(1,2), dropout=0.4) #80x80
		x = self.module_conv(x, 128, 128, dropout=0.3) #40x40
		x = self.module_conv(x, 256, 256, dropout=0.2) #20x20
		x = self.module_conv(x, 512, 512, dropout=0.1) #10x10
		x = self.module_conv(x, 1024, 1024) #5x5
		x = self.module_conv(x, 1024, 1024, s1=(1,1), p1="valid", p2="valid") #1x1
		self.model_image_o_image_enc = layers.Flatten()(x)

		self.model_image = keras.Model(
			inputs=self.model_image_i_image,
			outputs=self.model_image_o_image_enc,
			name="model_image")
		self.model_image.summary()

	def create_state_model(self):
		self.model_state_i_state = keras.Input(shape=(self.state_size))
		self.model_state_i_image_enc = keras.Input(shape=(self.image_enc_size))
		
		# concatenate image encoding and layer 
		x = layers.concatenate([self.model_state_i_state, self.model_state_i_image_enc])

		x = self.module_dense(x, 2048, dropout=0.5)
		x = self.module_dense(x, 1024, dropout=0.3)
		x = self.module_dense(x, 1024, dropout=0.1)
		x = self.module_dense(x, 1024)
		y = self.module_dense(x, self.state_size)

		# state output
		self.model_state_o_state =\
			layers.Dense(self.state_size,  kernel_initializer=self.initializer,
			activation="tanh")(y)
		
		x = self.module_dense(x, self.state_size)

		# gate output value for previous state feedthrough
		self.model_state_o_gate_prev =\
			layers.Dense(self.state_size, kernel_initializer=self.initializer,
			activation="sigmoid")(x)

		# gate output value for new state (1 - model_state_o_gate_prev)
		self.model_state_o_gate_new = layers.Lambda(lambda x: 1.0 - x)(
			self.model_state_o_gate_prev)

		self.model_state = keras.Model(
			inputs=[self.model_state_i_state, self.model_state_i_image_enc],
			outputs=[self.model_state_o_state, self.model_state_o_gate_prev,
				self.model_state_o_gate_new],
			name="model_state")
		self.model_state.summary()

	def create_action_model(self):
		self.model_action_i_state = keras.Input(shape=(self.state_size))

		x = self.module_dense(self.model_action_i_state, 1024, dropout=0.5)
		x = self.module_dense(x, 1024, dropout=0.3)
		x = self.module_dense(x, 512, dropout=0.1)
		x = self.module_dense(x, 256)
		x = self.module_dense(x, 128)

		self.model_action_o_action = layers.Dense(15, kernel_initializer=self.initializer,\
			activation="tanh")(x)

		self.model_action = keras.Model(
			inputs=self.model_action_i_state,
			outputs=self.model_action_o_action,
			name="model_action")
		self.model_action.summary()
	
	def create_forward_model(self):
		self.model_forward_i_state = keras.Input(shape=(self.state_size))
		self.model_forward_i_action = keras.Input(shape=(15))

		x = layers.concatenate([self.model_forward_i_state, self.model_forward_i_action])

		x = self.module_dense(x, self.state_size*2, dropout=0.5)
		x = self.module_dense(x, self.state_size, dropout=0.3)
		x = self.module_dense(x, self.state_size, dropout=0.1)

		self.model_forward_o_state =\
			layers.Dense(self.state_size,  kernel_initializer=self.initializer,\
			activation="tanh")(x)

		self.model_forward = keras.Model(
			inputs=[self.model_forward_i_state, self.model_forward_i_action],
			outputs=self.model_forward_o_state,
			name="model_forward")
		self.model_forward.summary()
	
	def create_inverse_model(self):
		self.model_inverse_i_state1 = keras.Input(shape=(self.state_size))
		self.model_inverse_i_state2 = keras.Input(shape=(self.state_size))

		x = layers.concatenate([self.model_inverse_i_state1, self.model_inverse_i_state2])

		x = self.module_dense(x, self.state_size*2, dropout=0.5)
		x = self.module_dense(x, self.state_size, dropout=0.3)
		x = self.module_dense(x, self.state_size, dropout=0.1)
		x = self.module_dense(x, 256)
		x = self.module_dense(x, 64)

		self.model_inverse_o_action = layers.Dense(15, kernel_initializer=self.initializer,
			activation="tanh")(x)

		self.model_inverse = keras.Model(
			inputs=[self.model_inverse_i_state1, self.model_inverse_i_state2],
			outputs=self.model_inverse_o_action,
			name="model_inverse")
		self.model_inverse.summary()
	
	def create_reward_model(self):
		self.model_reward_i_state = keras.Input(shape=(self.state_size))
		self.model_reward_i_action = keras.Input(shape=(15))

		x = layers.concatenate([self.model_reward_i_state, self.model_reward_i_action])

		x = self.module_dense(x, self.state_size, dropout=0.5)
		x = self.module_dense(x, self.state_size/2, dropout=0.3)
		x = self.module_dense(x, self.state_size/2, dropout=0.1)
		x = self.module_dense(x, 256)
		x = self.module_dense(x, 64)

		self.model_reward_o_reward =\
			layers.Dense(1,  kernel_initializer=self.initializer)(x)

		self.model_reward = keras.Model(
			inputs=[self.model_reward_i_state, self.model_reward_i_action],
			outputs=self.model_reward_o_reward,
			name="model_reward")
		self.model_reward.summary()
	
	def combine_model(self):
		image_enc = self.model_image(self.model_image_i_image)
		[state_new, gate_prev, gate_new] = self.model_state([self.model_state_i_state, image_enc])
		state = layers.Add()([\
			layers.Multiply()([self.model_state_i_state, gate_prev]),
			layers.Multiply()([state_new, gate_new])
		]) # new state mixed from old and new according to the gae value

		self.model_advance = keras.Model(
			inputs=[self.model_state_i_state, self.model_image_i_image],
			outputs=state)
		
		action_prev = self.model_inverse([self.model_state_i_state, state])
		#action = self.model_action([state]) # new action
		reward = self.model_reward([state, self.model_reward_i_action])

		self.model_combined = keras.Model(
			inputs=[
				self.model_state_i_state,
				self.model_image_i_image,
				self.model_reward_i_action],
			#outputs=[action_prev, action],
			outputs=[action_prev, reward],
			name="model_combined")
		self.model_combined.compile(
			loss="mean_squared_error",
			loss_weights=[1.0, 0.001],
			optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
		)

		self.model_forward.compile(
			loss="mean_squared_error",
			optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
		)

	def advance(self, image, action_prev):
		action_prev = convert_action_to_continuous(action_prev)
		self.state_prediction = self.model_forward(
			[np.expand_dims(self.state,0),
			np.expand_dims(action_prev,0)],
			training=False)[0]

		self.state = self.model_advance([
			np.expand_dims(self.state, 0),
			np.expand_dims(image, 0)],
			training=False)[0]
		
		# intrinsic model reward
		return (np.square(self.state_prediction - self.state)).mean()

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
		# action = self.model_action.predict(np.expand_dims(self.state,0))[0]
		# return convert_action_to_mixed(action)


		action = tf.expand_dims(tf.random.normal([15], mean=0.0, stddev=0.01), 0)
		# print(action.numpy())

		for i in range(10):
			with tf.GradientTape() as g:
				g.watch(action)
				reward = self.model_reward([np.expand_dims(self.state, 0), action], training=False)
			action_grad = g.gradient(reward, action)
			action_grad /= tf.math.reduce_std(action_grad) + 1e-8
			action = action + action_grad*self.action_predict_step_size

			#print(tf.math.reduce_max(tf.abs(action)).numpy())
			#print(reward[0])
		
		action_max = tf.math.reduce_max(tf.abs(action)).numpy()
		self.action_predict_step_size = ((1.0 / action_max)*0.1 + 0.9)*self.action_predict_step_size

		#print("action_max: {}".format(action_max))
		#print("step_size: {}".format(self.action_predict_step_size))
		#print("==========================================================================")

		print("B {:8.5f} ".format(reward[0].numpy()[0]), end="")
		return convert_action_to_mixed(action[0])

	def predict_worst_action(self):
		action = tf.expand_dims(tf.random.normal([15], mean=0.0, stddev=0.01), 0)

		for i in range(10):
			with tf.GradientTape() as g:
				g.watch(action)
				reward = self.model_reward([np.expand_dims(self.state, 0), action], training=False)
			action_grad = g.gradient(reward, action)
			action_grad /= tf.math.reduce_std(action_grad) + 1e-8
			action = action - action_grad*self.action_predict_step_size
		
		action_max = tf.math.reduce_max(tf.abs(action)).numpy()
		self.action_predict_step_size = ((1.0 / action_max)*0.1 + 0.9)*self.action_predict_step_size
		
		print("W {:8.5f} ".format(reward[0].numpy()[0]), end="")
		return convert_action_to_mixed(action[0])

	def train(self, state_prev, state, image, action_prev, action, reward):
		state_prev = np.asarray(state_prev)
		state = np.asarray(state)
		image = np.asarray(image)
		action_prev = np.asarray(action_prev)
		action = np.asarray(action)
		reward = np.asarray(reward)

		# first train the combined model
		# self.model_combined.fit(x=[state_prev, image], y=[action_prev, action],
		# 	batch_size=32, epochs=8, shuffle=True)
		self.model_combined.fit(x=[state_prev, image, action], y=[action_prev, reward],
			batch_size=64, epochs=1, shuffle=True)
		
		# then the forward model
		self.model_forward.fit(x=[state_prev, action_prev], y=state,
			batch_size=64, epochs=1, shuffle=True)

	def save_model(self, filename_prefix):
		self.model_image.save("{}_image.h5".format(filename_prefix))
		self.model_state.save("{}_state.h5".format(filename_prefix))
		self.model_action.save("{}_action.h5".format(filename_prefix))
		self.model_forward.save("{}_forward.h5".format(filename_prefix))
		self.model_inverse.save("{}_inverse.h5".format(filename_prefix))
		self.model_reward.save("{}_reward.h5".format(filename_prefix))
	
	def load_model(self, filename_prefix):
		self.model_image = keras.models.load_model("{}_image.h5".format(filename_prefix))
		self.model_state = keras.models.load_model("{}_state.h5".format(filename_prefix))
		self.model_action = keras.models.load_model("{}_action.h5".format(filename_prefix))
		self.model_forward = keras.models.load_model("{}_forward.h5".format(filename_prefix))
		self.model_inverse = keras.models.load_model("{}_inverse.h5".format(filename_prefix))
		self.model_reward = keras.models.load_model("{}_reward.h5".format(filename_prefix))

		self.combine_model()