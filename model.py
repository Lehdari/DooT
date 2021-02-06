import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import activations
import random
from utils import *


class Model:
	def __init__(self, episode_length, n_training_epochs):
		self.initializer = initializers.RandomNormal(stddev=0.02)
		self.optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
		self.loss_function = keras.losses.MeanSquaredError()

		self.state_size = 256
		self.image_enc_size = 256

		self.state = tf.zeros((self.state_size,))
		self.image_enc = tf.zeros((1, self.image_enc_size))
		self.action_predict_step_size = 0.01
		self.episode_length = episode_length
		self.n_training_epochs = n_training_epochs

		self.create_image_model(feature_multiplier=2)
		self.create_state_model()
		#self.create_action_model()
		self.create_forward_model()
		self.create_inverse_model()
		self.create_reward_model()
		self.create_recurrent_module()

	
	def module_dense(self, x, n, alpha=0.01, dropout=None):
		# if dropout is not None:
		# 	x = layers.Dropout(dropout)(x)

		use_shortcut = n == x.shape[1]

		if use_shortcut:
			y = x

		x = layers.Dense(n, kernel_initializer=self.initializer, use_bias=False)(x)
		x = layers.LayerNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
		x = activations.relu(x, alpha=alpha)
		# x = layers.PReLU(alpha_initializer=self.initializer)(x)

		if use_shortcut:
			return layers.Add()([x, y])
		else:
			return x
	
	def module_conv(self, x, n1, n2, k1=(3,3), k2=(3,3), s1=(2,2), s2=(1,1),
		p1="same", p2="same", bn2=True,
		alpha=0.01, dropout=None):

		# if dropout is not None:
		# 	x = layers.Dropout(dropout)(x)

		#shortcut by avg pooling
		pool_x_size = s1[0]*s2[0]
		pool_y_size = s1[1]*s2[1]
		if p1 == "valid":
			pool_x_size += k1[0]-1
			pool_y_size += k1[1]-1
		if p2 == "valid":
			pool_x_size += k2[0]-1
			pool_y_size += k2[1]-1
		y = layers.AveragePooling2D((pool_x_size, pool_y_size))(x)
		y = layers.Conv2D(n2, (1, 1), kernel_initializer=self.initializer, use_bias=False)(y)
		y = layers.LayerNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(y)
		#y = activations.relu(y, alpha=alpha)

		x = layers.Conv2D(n1, k1, padding=p1, kernel_initializer=self.initializer,
			strides=s1, use_bias=False)(x)
		x = layers.LayerNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
		x = activations.relu(x, alpha=alpha)
		# x = layers.PReLU(alpha_initializer=self.initializer)(x)

		if bn2:
			x = layers.Conv2D(n2, k2, padding=p2, kernel_initializer=self.initializer,
			strides=s2, use_bias=False)(x)
			x = layers.LayerNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
		else:
			x = layers.Conv2D(n2, k2, padding=p2, kernel_initializer=self.initializer,
			strides=s2)(x)
		x = activations.relu(x, alpha=alpha)
		# x = layers.PReLU(alpha_initializer=self.initializer)(x)

		return layers.Add()([x, y])
	
	def create_image_model(self, feature_multiplier=1):
		self.model_image_i_image = keras.Input(shape=(240, 320, 3))

		x = self.module_conv(self.model_image_i_image,
			8*feature_multiplier, 16*feature_multiplier,
			k1=(3,2), s1=(3,2), k2=(3,3), s2=(1,2), dropout=0.3) #80x80
		x = self.module_conv(x, 32*feature_multiplier, 32*feature_multiplier, dropout=0.3) #40x40
		x = self.module_conv(x, 64*feature_multiplier, 64*feature_multiplier, dropout=0.2) #20x20
		x = self.module_conv(x, 128*feature_multiplier, 128*feature_multiplier, dropout=0.1) #10x10
		x = self.module_conv(x, 256*feature_multiplier, 256*feature_multiplier, k2=(1,1)) #5x5
		x = self.module_conv(x, 256*feature_multiplier, 256*feature_multiplier,
			s1=(1,1), p1="valid", p2="valid", bn2=False) #1x1
		x = layers.Flatten()(x)
		# use regularizers for image encoding to avoid issue of output saturation
		self.model_image_o_image_enc = layers.Dense(self.image_enc_size,
			kernel_initializer=self.initializer,
			activity_regularizer=tf.keras.regularizers.l2(1.0/self.image_enc_size),
			activation="tanh")(x)

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
		x = self.module_dense(x, self.state_size*2, dropout=0.5)
		x = self.module_dense(x, self.state_size, dropout=0.3)

		# state output
		s = self.module_dense(x, self.state_size, dropout=0.1)
		s = layers.Dense(self.state_size,  kernel_initializer=self.initializer,
			activity_regularizer=tf.keras.regularizers.l2(1.0/self.state_size),
			activation="tanh")(s)
		
		# gate output value for previous state feedthrough
		g1 = self.module_dense(x, self.state_size, dropout=0.1)
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
		self.model_forward_i_image_enc = keras.Input(shape=(self.image_enc_size))
		self.model_forward_i_action = keras.Input(shape=(15))

		x = layers.concatenate([self.model_forward_i_image_enc, self.model_forward_i_action])

		x = self.module_dense(x, self.image_enc_size*2, dropout=0.5)
		x = self.module_dense(x, self.image_enc_size, dropout=0.3)
		x = self.module_dense(x, self.image_enc_size, dropout=0.1)

		self.model_forward_o_image_enc =\
			layers.Dense(self.image_enc_size,  kernel_initializer=self.initializer,\
			activation="tanh")(x)

		self.model_forward = keras.Model(
			inputs=[self.model_forward_i_image_enc, self.model_forward_i_action],
			outputs=self.model_forward_o_image_enc,
			name="model_forward")
		self.model_forward.summary()
		#self.model_forward.compile(loss=self.loss_function, optimizer=self.optimizer)
	
	def create_inverse_model(self):
		self.model_inverse_i_image_enc1 = keras.Input(shape=(self.image_enc_size))
		self.model_inverse_i_image_enc2 = keras.Input(shape=(self.image_enc_size))

		#x = layers.concatenate([self.model_inverse_i_image_enc1, self.model_inverse_i_image_enc2])
		x = layers.Subtract()([self.model_inverse_i_image_enc1, self.model_inverse_i_image_enc2])

		x = self.module_dense(x, self.image_enc_size, dropout=0.5)
		x = self.module_dense(x, self.image_enc_size, dropout=0.3)
		x = self.module_dense(x, self.image_enc_size, dropout=0.1)
		x = self.module_dense(x, self.image_enc_size/2)
		if int(self.image_enc_size/4) >= 64:
			x = self.module_dense(x, self.image_enc_size/4)

		self.model_inverse_o_action = layers.Dense(15, kernel_initializer=self.initializer,
			activation="tanh")(x)

		self.model_inverse = keras.Model(
			inputs=[self.model_inverse_i_image_enc1, self.model_inverse_i_image_enc2],
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

		y = self.module_dense(x, self.state_size/4)
		if int(self.image_enc_size/8) >= 64:
			y = self.module_dense(y, self.image_enc_size/8)
		# state step reward
		self.model_reward_o_reward_step = layers.Dense(1,  kernel_initializer=self.initializer)(y)
		
		x = self.module_dense(x, self.state_size/4)
		if int(self.image_enc_size/8) >= 64:
			x = self.module_dense(x, self.image_enc_size/8)
		# average reward
		self.model_reward_o_reward_avg = layers.Dense(1,  kernel_initializer=self.initializer)(x)

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

		self.recurrent_module_o_image_enc = self.model_image(self.model_image_i_image)

		self.recurrent_module_o_state = self.model_state([
			self.model_state_i_state, self.recurrent_module_o_image_enc])
		
		self.recurrent_module_o_action_prev = self.model_inverse([
			self.model_state_i_state, self.recurrent_module_o_state])

		self.recurrent_module_o_reward_step, self.recurrent_module_o_reward_avg =\
			self.model_reward([self.recurrent_module_o_state, self.model_reward_i_action])

		self.recurrent_module_o_reward_sum = layers.Add()([
			self.recurrent_module_i_reward_sum, self.recurrent_module_o_reward_step])

		self.recurrent_module = keras.Model(
			inputs=[
				self.model_state_i_state,
				self.recurrent_module_i_reward_sum,
				#self.model_inverse_i_image_enc1, # encoding of previous image
				self.model_image_i_image,
				self.model_reward_i_action
			],
			outputs=[
				self.recurrent_module_o_state,
				self.recurrent_module_o_image_enc,
				self.recurrent_module_o_action_prev,
				self.recurrent_module_o_reward_sum,
				self.recurrent_module_o_reward_step,
				self.recurrent_module_o_reward_avg,
			],
			name="recurrent_module"
		)
		self.recurrent_module.summary()
		#self.recurrent_module.compile(loss=self.loss_function, optimizer=self.optimizer)

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

	def advance(self, image, action):
		action = convert_action_to_continuous(action)

		# image encoding prediction according to previous image encoding and the action
		image_enc_pred = self.model_forward([self.image_enc, tf.expand_dims(action, 0)],
			training=False)

		# update the image encoding for next advance step
		self.image_enc = self.model_image(tf.expand_dims(image, 0), training=False)

		self.state = self.model_state(
			[tf.expand_dims(self.state,0), self.image_enc], training=False)[0]
		
		# intrinsic model reward
		return (np.square(image_enc_pred - self.image_enc)).mean()

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

	def train_subsequence(self,
		image, action, reward_step, reward_avg,
		begin, end,
		state, reward_sum, image_enc):
		g_recurrent_module = []
		g_model_forward = []

		n = end-begin
		state_begin = state
		reward_sum_begin = reward_sum

		with tf.GradientTape() as gt:
			for i in range(begin, end):
				if i > 0:
					action_prev = action[i-1:i]
				else:
					action_prev = tf.zeros(shape=(1,15))

				image_enc_prev = image_enc
				state_prev = state
				state, image_enc, action_prev_pred, reward_sum, reward_step_pred, reward_avg_pred =\
					self.recurrent_module([state, reward_sum, image[i:(i+1)],
					action[i:(i+1)]], training=True)
				
			loss_reward = self.loss_function(reward_step[i], reward_step_pred)
			loss_reward += self.loss_function(reward_avg[i], reward_avg_pred)
			loss_action = self.loss_function(action_prev, action_prev_pred)
			loss_img_enc = self.model_image.losses[0]
			loss_state = self.model_state.losses[0]

			#loss = loss_reward + loss_action
			loss = loss_reward*1.0e-10 + loss_action + loss_img_enc*0.01 + loss_state*0.01
			self.loss_reward_sum += loss_reward.numpy()
			self.loss_action_sum += loss_action.numpy()
			self.loss_sum += loss.numpy()

				# if i == 0:
				# 	g_recurrent_module = gt.gradient(loss, self.recurrent_module.trainable_variables)
				# else:
				# 	gg = gt.gradient(loss, self.recurrent_module.trainable_variables)
				# 	for j in range(len(g_recurrent_module)):
				# 		g_recurrent_module[j] += gg[j]

			# with tf.GradientTape() as gt:
			# 	state_pred = self.model_forward([state_prev, action_prev], training=True)
			# 	loss_forward = self.loss_function(state, state_pred)
			# 	self.loss_forward_sum += loss_forward.numpy()

			# 	if i == 0:
			# 		g_model_forward = gt.gradient(loss_forward,
			# 			self.model_forward.trainable_variables)
			# 	else:
			# 		gg = gt.gradient(loss_forward, self.model_forward.trainable_variables)
			# 		for j in range(len(g_model_forward)):
			# 			g_model_forward[j] += gg[j]\
			
			print("{} / {}".format(i+1, self.episode_length), end="\r")
		
		g_recurrent_module = gt.gradient(loss, self.recurrent_module.trainable_variables)

		self.optimizer.apply_gradients(zip(g_recurrent_module,
			self.recurrent_module.trainable_variables))

		self.optimizer.apply_gradients(zip(g_model_forward,
			self.model_forward.trainable_variables))
		
		state, image_enc, action_prev_pred, reward_sum, reward_step_pred, reward_avg_pred =\
			self.recurrent_module([state_begin, reward_sum_begin, image[begin:(begin+1)],
			action[begin:(begin+1)]], training=False)

		return state, reward_sum, image_enc

	def train(self, image, action, reward_step, reward_avg):
		sequence_length = 4

		self.loss_action_avg = 1.0
		e = 0
		while self.loss_action_avg > 0.001 and e < self.n_training_epochs:
			self.loss_reward_sum = 0.0
			self.loss_action_sum = 0.0
			self.loss_forward_sum = 0.0
			self.loss_sum = 0.0

			state = tf.zeros(shape=(1,self.state_size))
			reward_sum = tf.zeros(shape=(1,))
			image_enc = tf.zeros(shape=(1,self.image_enc_size))

			for i in range(self.episode_length - sequence_length + 1):
				state, reward_sum, image_enc = self.train_subsequence(
					image, action, reward_step, reward_avg,
					i, i+sequence_length, state, reward_sum, image_enc)

			print("epoch {:2d}: l_reward: {:8.5f}, l_action: {:8.5f} l_total: {:8.5f} l_forward: {:10.9f}".format(
				e, self.loss_reward_sum / self.episode_length,
				self.loss_action_sum / self.episode_length,
				self.loss_sum / self.episode_length,
				self.loss_forward_sum / self.episode_length))
			
			e += 1
		
			self.loss_action_avg = self.loss_action_sum / self.episode_length

	def save_model(self, filename_prefix):
		self.model_image.save("{}_image.h5".format(filename_prefix))
		self.model_state.save("{}_state.h5".format(filename_prefix))
		#self.model_action.save("{}_action.h5".format(filename_prefix))
		self.model_forward.save("{}_forward.h5".format(filename_prefix))
		self.model_inverse.save("{}_inverse.h5".format(filename_prefix))
		self.model_reward.save("{}_reward.h5".format(filename_prefix))
	
	def load_model(self, filename_prefix):
		self.model_image = keras.models.load_model("{}_image.h5".format(filename_prefix))
		self.model_state = keras.models.load_model("{}_state.h5".format(filename_prefix))
		#self.model_action = keras.models.load_model("{}_action.h5".format(filename_prefix))
		self.model_forward = keras.models.load_model("{}_forward.h5".format(filename_prefix))
		self.model_inverse = keras.models.load_model("{}_inverse.h5".format(filename_prefix))
		self.model_reward = keras.models.load_model("{}_reward.h5".format(filename_prefix))

		self.create_recurrent_module()