import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import activations
import random
from utils import *


class Model:
	def __init__(self, episode_length):
		self.initializer = initializers.RandomNormal(stddev=0.02)
		#self.optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.75, nesterov=True)
		self.optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
		self.loss_function = keras.losses.MeanSquaredError()

		self.state_size = 1024
		self.image_enc_size = 1024

		self.state = tf.zeros((self.state_size,))
		self.action_predict_step_size = 0.01
		self.episode_length = episode_length

		self.create_image_model()
		self.create_state_model()
		self.create_action_model()
		self.create_forward_model()
		self.create_inverse_model()
		self.create_reward_model()
		self.create_recurrent_module()

		#self.create_sequence_model(episode_length)

	
	def module_dense(self, x, n, alpha=0.01, dropout=None):
		x = layers.Dense(n, kernel_initializer=self.initializer)(x)
		x = activations.relu(x, alpha=alpha)

		#x = layers.BatchNormalization(axis=-1)(x)

		if dropout is not None:
			x = layers.Dropout(dropout)(x)
		
		return x
	
	def module_conv(self, x, n1, n2, k1=(3,3), k2=(3,3), s1=(2,2), s2=(1,1),
		p1="same", p2="same", a1="relu", a2="relu", dropout=None):

		x = layers.Conv2D(n1, k1, padding=p1, kernel_initializer=self.initializer,
			strides=s1, activation="relu")(x)
		#x = layers.BatchNormalization(axis=-1)(x)

		x = layers.Conv2D(n2, k2, padding=p2, kernel_initializer=self.initializer,
			strides=s2, activation="relu")(x)
		#x = layers.BatchNormalization(axis=-1)(x)

		if dropout is not None:
			x = layers.Dropout(dropout)(x)
		
		return x
	
	def create_image_model(self):
		self.model_image_i_image = keras.Input(shape=(240, 320, 3))

		x = self.module_conv(self.model_image_i_image, 32, 64,
			k1=(3,2), s1=(3,2), k2=(3,3), s2=(1,2), dropout=0.4) #80x80
		x = self.module_conv(x, 128, 128, dropout=0.3) #40x40
		x = self.module_conv(x, 256, 256, dropout=0.2) #20x20
		x = self.module_conv(x, 512, 512, dropout=0.1) #10x10
		x = self.module_conv(x, 1024, 1024) #5x5
		x = self.module_conv(x, 1024, 1024, s1=(1,1), p1="valid", p2="valid", a2="linear") #1x1
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

		# intermediate layers
		x = self.module_dense(x, 2048, dropout=0.5)
		x = self.module_dense(x, 1024, dropout=0.25)

		# state output
		s = self.module_dense(x, self.state_size)
		s = layers.Dense(self.state_size,  kernel_initializer=self.initializer,
			activation="tanh")(s)
		
		# gate output value for previous state feedthrough
		g1 = self.module_dense(x, self.state_size)
		g1 = layers.Dense(self.state_size, kernel_initializer=self.initializer,
			activation="sigmoid")(g1)

		# gate output value for new state (1 - g1)
		g2 = layers.Lambda(lambda x: 1.0 - x)(g1)

		# gated new state
		self.model_state_o_state = layers.Add()([\
			layers.Multiply()([self.model_state_i_state, g1]),
			layers.Multiply()([s, g2])
		]) # new state mixed from old and new according to the gate values

		self.model_state = keras.Model(
			inputs=[self.model_state_i_state, self.model_state_i_image_enc],
			outputs=self.model_state_o_state,
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

		y = self.module_dense(x, 256)
		y = self.module_dense(y, 64)
		# state step reward
		self.model_reward_o_reward_step = layers.Dense(1,  kernel_initializer=self.initializer)(y)
		
		x = self.module_dense(x, 256)
		x = self.module_dense(x, 64)
		# average reward
		self.model_reward_o_reward_avg = layers.Dense(1,  kernel_initializer=self.initializer)(y)

		self.model_reward = keras.Model(
			inputs=[self.model_reward_i_state, self.model_reward_i_action],
			outputs=[self.model_reward_o_reward_step, self.model_reward_o_reward_avg],
			name="model_reward")
		self.model_reward.summary()
	'''
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
	'''
	def create_recurrent_module(self):
		self.recurrent_module_i_reward_sum = keras.Input(shape=(1))
		image_enc = self.model_image(self.model_image_i_image)

		self.recurrent_module_o_state = self.model_state([self.model_state_i_state, image_enc])
		self.recurrent_module_o_reward_step, self.recurrent_module_o_reward_avg =\
			self.model_reward([self.recurrent_module_o_state, self.model_reward_i_action])
		self.recurrent_module_o_reward_sum = layers.Add()([
			self.recurrent_module_i_reward_sum, self.recurrent_module_o_reward_step])

		self.recurrent_module = keras.Model(
			inputs=[
				self.model_state_i_state,
				self.recurrent_module_i_reward_sum,
				self.model_image_i_image,
				self.model_reward_i_action
			],
			outputs=[
				self.recurrent_module_o_state,
				self.recurrent_module_o_reward_sum,
				self.recurrent_module_o_reward_step,
				self.recurrent_module_o_reward_avg
			],
			name="recurrent_module"
		)
		self.recurrent_module.summary()
		self.recurrent_module.compile(loss=self.loss_function, optimizer=self.optimizer)

	'''
	def create_sequence_model(self, episode_length):
		seq_state = tf.zeros(shape=(1,self.state_size))
		seq_reward_sum = tf.zeros(shape=(1,1))

		self.model_sequence_i_image = keras.Input(shape=(episode_length, 240, 320, 3))
		self.model_sequence_i_action = keras.Input(shape=(episode_length, 15))
		rewards_out = []
		for i in range(episode_length):
			#self.model_sequence_i_image.append(keras.Input(shape=(240, 320, 3)))
			#self.model_sequence_i_action.append(keras.Input(shape=(15)))
			seq_state, seq_reward, reward_step = self.recurrent_module(
				[seq_state, seq_reward_sum,
				self.model_sequence_i_image[:,i],
				self.model_sequence_i_action[:,i]])
			rewards_out.append(reward_step)
		self.model_sequence_o_reward = tf.concat(rewards_out, axis=1)
		
		self.model_sequence = keras.Model(
			inputs = [self.model_sequence_i_image, self.model_sequence_i_action],
			outputs = self.model_sequence_o_reward,
			name = "model_sequence"
		)
		self.model_sequence.summary()
		self.model_sequence.compile(
			loss="mean_squared_error",
			#optimizer=keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999)
			optimizer=keras.optimizers.SGD(learning_rate=0.00001, momentum=0.75, nesterov=True)
		)
'''

	def advance(self, image):
		new_state = self.model_state(
			[tf.expand_dims(self.state,0),
			self.model_image(tf.expand_dims(image, 0), training=False)],
			training=False)#[0]
		
		self.state = new_state[0]
		# intrinsic model reward
		#return (np.square(self.state_prediction - self.state)).mean()

	"""
	Reset state (after an episode)
	"""
	def reset_state(self):
		self.state = tf.zeros((self.state_size,))

	"""
	Predict action from the state of the model
	return: list length of 15: 14 booleans and 1 float
	"""
	def predict_action(self):
		step_reward_weight = 0.25

		action = tf.expand_dims(tf.random.normal([15], mean=0.0, stddev=0.01), 0)

		for i in range(10):
			with tf.GradientTape() as g:
				g.watch(action)
				reward_step, reward_avg = self.model_reward([np.expand_dims(self.state, 0), action],
				training=False)
				reward = reward_step*step_reward_weight + reward_avg
				#print("{:8.6f} {:8.6f} {:8.6f}".format(
				#	reward[0].numpy()[0], reward_step[0].numpy()[0], reward_avg[0].numpy()[0]))
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

		print("B {:8.5f} {:8.5f} ".format(reward_step[0].numpy()[0], reward_avg[0].numpy()[0]), end="")
		return convert_action_to_mixed(action[0])

	def predict_worst_action(self):
		step_reward_weight = 0.25

		action = tf.expand_dims(tf.random.normal([15], mean=0.0, stddev=0.01), 0)

		for i in range(10):
			with tf.GradientTape() as g:
				g.watch(action)
				reward_step, reward_avg = self.model_reward([np.expand_dims(self.state, 0), action],
				training=False)
				reward = reward_step*step_reward_weight + reward_avg
			action_grad = g.gradient(reward, action)
			action_grad /= tf.math.reduce_std(action_grad) + 1e-8
			action = action - action_grad*self.action_predict_step_size
		
		action_max = tf.math.reduce_max(tf.abs(action)).numpy()
		self.action_predict_step_size = ((1.0 / action_max)*0.1 + 0.9)*self.action_predict_step_size
		
		print("W {:8.5f} {:8.5f} ".format(reward_step[0].numpy()[0], reward_avg[0].numpy()[0]), end="")
		return convert_action_to_mixed(action[0])

	def train(self, image, action, reward_step, reward_avg):
		#image = tf.convert_to_tensor(image)
		#action = tf.convert_to_tensor(action)
		#reward = tf.convert_to_tensor(reward)

		n_training_epochs = 4

		for e in range(n_training_epochs):
			state = tf.zeros(shape=(1,self.state_size))
			reward_sum = tf.zeros(shape=(1,))
			loss = 0.0
			g = []
			for i in range(self.episode_length):
				with tf.GradientTape() as gt:
					state, reward_sum, reward_step_pred, reward_avg_pred =\
						self.recurrent_module([state, reward_sum, image[i:(i+1)], action[i:(i+1)]])
					l = self.loss_function(reward_step[i], reward_step_pred)
					l += self.loss_function(reward_avg[i], reward_avg_pred)
					loss += l

					if i == 0:
						g = gt.gradient(l, self.recurrent_module.trainable_variables)
					else:
						gg = gt.gradient(l, self.recurrent_module.trainable_variables)
						for j in range(len(g)):
							g[j] += gg[j]

					print("{} / {}".format(i+1, self.episode_length), end="\r")
			print("epoch {:2d} loss:   {}".format(e, loss / self.episode_length))

			for i in range(len(g)):
				g[i] /= self.episode_length
			self.optimizer.apply_gradients(zip(g, self.recurrent_module.trainable_variables))

		state = tf.zeros(shape=(1,self.state_size))
		reward_sum = tf.zeros(shape=(1,))
		loss = 0.0
		for i in range(self.episode_length):
			state, reward_sum, reward_step_pred, reward_avg_pred =\
				self.recurrent_module([state, reward_sum, image[i:(i+1)], action[i:(i+1)]])
			loss += self.loss_function(reward_step[i], reward_step_pred)
			loss += self.loss_function(reward_avg[i], reward_avg_pred)
			print("{} / {}".format(i+1, self.episode_length), end="\r")

		print("post-train loss: {}".format(loss / self.episode_length))

		# first train the combined model
		# self.model_sequence.train_on_batch(
		# 	[image, action],
		# 	reward
		# )
		#self.model_sequence.fit(x=[image, action], y=reward,
		#	batch_size=1, epochs=16, shuffle=False)
		
		# then the forward model
		# self.model_forward.fit(x=[state_prev, action_prev], y=state,
		# 	batch_size=64, epochs=1, shuffle=True)

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

		self.create_recurrent_module()