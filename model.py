import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras import regularizers
import random
from utils import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import cv2


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class L2Regularizer(regularizers.Regularizer):
	def __init__(self, strength):
		self.strength = tf.Variable(strength)

	@tf.function
	def __call__(self, x):
		return self.strength * tf.reduce_mean(tf.square(x))

def image_loss(y_true, y_pred):
	# gx_true = y_true[:,:,1:,:]-y_true[:,:,0:-1,:]
	# gx_pred = y_pred[:,:,1:,:]-y_pred[:,:,0:-1,:]
	# gy_true = y_true[:,1:,:,:]-y_true[:,0:-1,:,:]
	# gy_pred = y_pred[:,1:,:,:]-y_pred[:,0:-1,:,:]

	# gx_true2 = y_true[:,:,1::2,:]-y_true[:,:,0:-1:2,:]
	# gx_pred2 = y_pred[:,:,1::2,:]-y_pred[:,:,0:-1:2,:]
	# gy_true2 = y_true[:,1::2,:,:]-y_true[:,0:-1:2,:,:]
	# gy_pred2 = y_pred[:,1::2,:,:]-y_pred[:,0:-1:2,:,:]

	# gx_true4 = y_true[:,:,4::4,:]-y_true[:,:,0:-4:4,:]
	# gx_pred4 = y_pred[:,:,4::4,:]-y_pred[:,:,0:-4:4,:]
	# gy_true4 = y_true[:,4::4,:,:]-y_true[:,0:-4:4,:,:]
	# gy_pred4 = y_pred[:,4::4,:,:]-y_pred[:,0:-4:4,:,:]

	# gx_true8 = y_true[:,:,8::8,:]-y_true[:,:,0:-8:8,:]
	# gx_pred8 = y_pred[:,:,8::8,:]-y_pred[:,:,0:-8:8,:]
	# gy_true8 = y_true[:,8::8,:,:]-y_true[:,0:-8:8,:,:]
	# gy_pred8 = y_pred[:,8::8,:,:]-y_pred[:,0:-8:8,:,:]

	# gx_true16 = y_true[:,:,16::16,:]-y_true[:,:,0:-16:16,:]
	# gx_pred16 = y_pred[:,:,16::16,:]-y_pred[:,:,0:-16:16,:]
	# gy_true16 = y_true[:,16::16,:,:]-y_true[:,0:-16:16,:,:]
	# gy_pred16 = y_pred[:,16::16,:,:]-y_pred[:,0:-16:16,:,:]

	return tf.reduce_mean(tf.abs(y_true - y_pred))# +\
		#2.0*(tf.reduce_mean(tf.abs(gx_true16-gx_pred16)) + tf.reduce_mean(tf.abs(gy_true16-gy_pred16))) +\
		#4.0*(tf.reduce_mean(tf.abs(gx_true8-gx_pred8)) + tf.reduce_mean(tf.abs(gy_true8-gy_pred8))) +\
		#8.0*(tf.reduce_mean(tf.abs(gx_true4-gx_pred4)) + tf.reduce_mean(tf.abs(gy_true4-gy_pred4))) +\
		#16.0*(tf.reduce_mean(tf.abs(gx_true2-gx_pred2)) + tf.reduce_mean(tf.abs(gy_true2-gy_pred2))) +\
		#32.0*(tf.reduce_mean(tf.abs(gx_true-gx_pred)) + tf.reduce_mean(tf.abs(gy_true-gy_pred)))
		


class Model:
	def __init__(self, episode_length, n_training_epochs):
		self.initializer = initializers.RandomNormal(stddev=0.02)
		self.optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
		self.loss_function = keras.losses.MeanSquaredError()
		self.loss_image = image_loss

		self.state_size = 256
		self.image_enc_size = 256

		self.state = tf.zeros((self.state_size,))
		self.image_enc = tf.zeros((1, self.image_enc_size))
		self.action_predict_step_size = 0.01
		self.episode_length = episode_length
		self.n_training_epochs = n_training_epochs

		self.create_image_encoder_model(feature_multiplier=1)
		self.create_image_decoder_model(feature_multiplier=1)

		#self.create_state_model()
		#self.create_action_model()
		#self.create_forward_model()
		#self.create_inverse_model()
		#self.create_reward_model()
		#self.create_recurrent_module()

	
	def module_dense(self, x, n, x2=None, n2=None, alpha=0.001, act=None):
		use_shortcut = n == x.shape[1]

		if use_shortcut:
			y = x

		# enable concatenation if auxiliary input is provided
		if x2 is not None:
			x = layers.Concatenate()([x, x2])
		
		# double layer model
		if n2 is not None:
			x = layers.Dense(n2, kernel_initializer=self.initializer, use_bias=False,
				kernel_regularizer=L2Regularizer(1.0e-6))(x)
			x = layers.BatchNormalization(axis=-1,
				beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
				gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
			x = activations.relu(x, alpha=alpha)

		x = layers.Dense(n, kernel_initializer=self.initializer, use_bias=False,
			kernel_regularizer=L2Regularizer(1.0e-6))(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
		x = activations.relu(x, alpha=alpha)

		if use_shortcut:
			x = layers.Add()([x, y])
		
		if act is not None:
			x = act(x)
		
		return x
		
	
	def module_conv(self, x, n1, n2, k1=(3,3), k2=(3,3), s1=(2,2), s2=(1,1),
		p1="same", p2="same", alpha=0.001):

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
		y = layers.Conv2D(n2, (1, 1), kernel_initializer=self.initializer, use_bias=False,
			kernel_regularizer=L2Regularizer(5.0e-4))(y)
		y = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(y)

		x = layers.Conv2D(n1, k1, padding=p1, kernel_initializer=self.initializer,
			strides=s1, use_bias=False,
			kernel_regularizer=L2Regularizer(5.0e-4))(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
		x = activations.relu(x, alpha=alpha)

		x = layers.Conv2D(n2, k2, padding=p2, kernel_initializer=self.initializer,
			strides=s2, use_bias=False,
			kernel_regularizer=L2Regularizer(5.0e-4))(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
		x = activations.relu(x, alpha=alpha)

		return layers.Add()([x, y])
	

	def module_deconv(self, x, n1, n2, k1=(4,4), k2=(2,2), s1=(2,2), s2=(1,1),
		p1="same", p2="same", bn2=True, alpha=0.001, act=None):

		#shortcut by linear upsampling
		pool_x_size = s1[0]*s2[0]
		pool_y_size = s1[1]*s2[1]
		if p1 == "valid":
			pool_x_size += k1[0]-1
			pool_y_size += k1[1]-1
		if p2 == "valid":
			pool_x_size += k2[0]-1
			pool_y_size += k2[1]-1
		y = layers.UpSampling2D((pool_x_size, pool_y_size), interpolation="bilinear")(x)
		y = layers.Conv2D(n2, (1, 1), kernel_initializer=self.initializer, use_bias=False,
			kernel_regularizer=L2Regularizer(5.0e-4))(y)
		y = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(y)

		x = layers.Conv2DTranspose(n1, k1, padding=p1, kernel_initializer=self.initializer,
			strides=s1, use_bias=False,
			kernel_regularizer=L2Regularizer(5.0e-4))(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
		x = activations.relu(x, alpha=alpha)

		x = layers.Conv2DTranspose(n2, k2, padding=p2, kernel_initializer=self.initializer,
			strides=s2, use_bias=False,
			kernel_regularizer=L2Regularizer(5.0e-4))(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
			gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1))(x)
		x = activations.relu(x, alpha=alpha)

		x = layers.Add()([x, y])

		if act is not None:
			x = act(x)
		
		return x
	

	def create_image_encoder_model(self, feature_multiplier=1):
		self.model_image_encoder_i_image = keras.Input(shape=(240, 320, 3))

		x = self.module_conv(self.model_image_encoder_i_image,
			8*feature_multiplier, 16*feature_multiplier,
			k1=(3,2), s1=(3,2), k2=(3,3), s2=(1,2)) #80x80
		x = self.module_conv(x, 32*feature_multiplier, 32*feature_multiplier) #40x40
		x = self.module_conv(x, 64*feature_multiplier, 64*feature_multiplier) #20x20
		x = self.module_conv(x, 128*feature_multiplier, 128*feature_multiplier) #10x10
		x = self.module_conv(x, 256*feature_multiplier, 256*feature_multiplier, k2=(1,1)) #5x5
		x = self.module_conv(x, 256*feature_multiplier, 256*feature_multiplier,
			s1=(1,1), p1="valid", p2="valid") #1x1
		x = layers.Flatten()(x)
		self.model_image_encoder_o_image_enc = self.module_dense(x, self.image_enc_size,
			act=layers.Activation(activations.tanh))

		self.model_image_encoder = keras.Model(
			inputs=self.model_image_encoder_i_image,
			outputs=self.model_image_encoder_o_image_enc,
			name="model_image_encoder")
		self.model_image_encoder.summary()
	

	def create_image_decoder_model(self, feature_multiplier=1):
		self.model_image_decoder_i_image_enc = keras.Input(shape=(self.image_enc_size))
		x = layers.Reshape((1, 1, -1))(self.model_image_decoder_i_image_enc)

		x = self.module_deconv(x, 256*feature_multiplier, 256*feature_multiplier,
			k1=(3,3), s1=(1,1), k2=(3,3), s2=(1,1), p1="valid", p2="valid", alpha=1.0e-6) #5x5
		x = self.module_deconv(x, 256*feature_multiplier, 128*feature_multiplier,
			k1=(3,4), s1=(3,4), k2=(3,3), s2=(1,1), alpha=1.0e-6) #20x15
		x = self.module_deconv(x, 128*feature_multiplier, 64*feature_multiplier, alpha=1.0e-6) #40x30
		x = self.module_deconv(x, 64*feature_multiplier, 32*feature_multiplier, alpha=1.0e-6) #80x60
		x = self.module_deconv(x, 32*feature_multiplier, 16*feature_multiplier, k2=(3,3), alpha=1.0e-6) #160x120

		self.model_image_decoder_o_image = self.module_deconv(x, 16*feature_multiplier, 3,
			act=layers.Activation(activations.sigmoid), k2=(3,3), alpha=1.0e-6)

		self.model_image_decoder = keras.Model(
			inputs=self.model_image_decoder_i_image_enc,
			outputs=self.model_image_decoder_o_image,
			name="model_image_decoder")
		self.model_image_decoder.summary()

	def create_state_model(self):
		self.model_state_i_state = keras.Input(shape=(self.state_size))
		self.model_state_i_image_enc = keras.Input(shape=(self.image_enc_size))
		
		# concatenate image encoding and layer 
		#x = layers.concatenate([self.model_state_i_state, self.model_state_i_image_enc])

		# first layer uses concatenative dense module
		x = self.module_dense(self.model_state_i_state, self.state_size,
			x2=self.model_state_i_image_enc, n2=self.state_size+self.image_enc_size)
		#x = self.module_dense(x, self.state_size)

		# state output
		s = self.module_dense(x, self.state_size)
		s = layers.Dense(self.state_size,  kernel_initializer=self.initializer,
			activity_regularizer=L2Regularizer(1.0),
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

	def create_forward_model(self):
		self.model_forward_i_state = keras.Input(shape=(self.state_size))
		self.model_forward_i_action = keras.Input(shape=(15))

		x = layers.concatenate([self.model_forward_i_state, self.model_forward_i_action])

		x = self.module_dense(x, self.state_size, n2=self.state_size)

		self.model_forward_o_state =\
			layers.Dense(self.state_size,  kernel_initializer=self.initializer,\
			activation="tanh")(x)

		self.model_forward = keras.Model(
			inputs=[self.model_forward_i_state, self.model_forward_i_action],
			outputs=self.model_forward_o_state,
			name="model_forward")
		self.model_forward.summary()
	
	def create_inverse_model(self):
		self.model_inverse_i_states = keras.Input(shape=(self.state_size, 1, 9))

		x = self.module_conv(self.model_inverse_i_states, 16, 2, k1=(1,1), k2=(1,1), s1=(1,1))
		x = layers.Flatten()(x)

		x = self.module_dense(x, self.state_size, n2=self.state_size*2)
		x = self.module_dense(x, self.state_size/2)
		if int(self.state_size/4) >= 64:
			x = self.module_dense(x, self.state_size/4)

		self.model_inverse_o_action = layers.Dense(15, kernel_initializer=self.initializer,
			activation="tanh")(x)

		self.model_inverse = keras.Model(
			inputs=self.model_inverse_i_states,
			outputs=self.model_inverse_o_action,
			name="model_inverse")
		self.model_inverse.summary()
	
	def create_reward_model(self):
		self.model_reward_i_state = keras.Input(shape=(self.state_size))
		self.model_reward_i_action = keras.Input(shape=(15))

		x = layers.concatenate([self.model_reward_i_state, self.model_reward_i_action])

		# x = self.module_dense(x, self.image_enc_size, n2=self.image_enc_size)

		y = self.module_dense(x, self.state_size/2)
		if int(self.image_enc_size/4) >= 64:
			y = self.module_dense(y, self.image_enc_size/4)
		# state step reward
		self.model_reward_o_reward_step = layers.Dense(1,  kernel_initializer=self.initializer)(y)
		
		x = self.module_dense(x, self.state_size/2)
		if int(self.image_enc_size/4) >= 64:
			x = self.module_dense(x, self.image_enc_size/4)
		# average reward
		self.model_reward_o_reward_avg = layers.Dense(1,  kernel_initializer=self.initializer)(x)

		self.model_reward = keras.Model(
			inputs=[self.model_reward_i_state, self.model_reward_i_action],
			outputs=[self.model_reward_o_reward_step, self.model_reward_o_reward_avg],
			name="model_reward")
		self.model_reward.summary()

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
		begin, end, state):
		n = end-begin
		state_begin = state

		with tf.GradientTape(persistent=True) as gt:
			states = [state[0]]
			for i in range(begin, end):
				#state_prev = state
				
				image_enc = self.model_image(image[i:(i+1)], training=True)
				state = self.model_state([state, image_enc], training=True)
				states.append(state[0])
			
			action_pred = self.model_inverse(tf.expand_dims(
				tf.expand_dims(tf.stack(states, axis=-1), axis=1), axis=0), training=True)

			# if end > 1:
			# 	action_prev = action[end-2:end-1]
			# else:
			# 	action_prev = tf.zeros(shape=(1,15))

			#loss_reward = self.loss_function(reward_step[i], reward_step_pred)
			#loss_reward += self.loss_function(reward_avg[i], reward_avg_pred)
			#loss_action = self.loss_function(action_prev, action_prev_pred)
			loss_action = self.loss_function(action[begin], action_pred)
			loss_model_image = tf.reduce_sum(self.model_image.losses)#*self.weight_loss_model_image
			loss_model_state = tf.reduce_sum(self.model_state.losses)#*self.weight_loss_model_state

			loss = loss_action + loss_model_image + loss_model_state
			
			self.loss_action_sum += loss_action#.numpy()
			self.loss_sum += loss#.numpy()
			self.loss_n += 1

			print("I {} / {}".format(i+1, self.episode_length), end="\r")
		
		if self.g_model_image == None:
			self.g_model_image = gt.gradient(loss, self.model_image.trainable_variables)
		else:
			gg = gt.gradient(loss, self.model_image.trainable_variables)
			for i in range(len(self.g_model_image)):
				self.g_model_image[i] += gg[i]
		
		if self.g_model_state == None:
			self.g_model_state = gt.gradient(loss, self.model_state.trainable_variables)
		else:
			gg = gt.gradient(loss, self.model_state.trainable_variables)
			for i in range(len(self.g_model_state)):
				self.g_model_state[i] += gg[i]
		
		if self.g_model_inverse == None:
			self.g_model_inverse = gt.gradient(loss, self.model_inverse.trainable_variables)
		else:
			gg = gt.gradient(loss, self.model_inverse.trainable_variables)
			for i in range(len(self.g_model_inverse)):
				self.g_model_inverse[i] += gg[i]
		
		state = self.model_state([state_begin,
			self.model_image(image[begin:(begin+1)],
			training=False)], training=False)

		return state

	def train(self, image, action, reward_step, reward_avg):
		sequence_length_image = 8
		sequence_length_state = 8

		self.loss_action_avg = 1.0
		e = 0
		while self.loss_action_avg > 0.001 and e < self.n_training_epochs:
			self.loss_reward_sum = 0.0
			self.loss_action_sum = 0.0
			self.loss_forward_sum = 0.0
			self.loss_sum = 0.0
			self.loss_n = 0

			self.g_model_image = None
			self.g_model_state = None
			self.g_model_inverse = None

			state = tf.zeros(shape=(1,self.state_size))
			for i in range(self.episode_length - sequence_length_image + 1):
				state = self.train_subsequence(image, action,
					reward_step, reward_avg, i, i+sequence_length_image, state)
			
			self.optimizer.apply_gradients(zip(self.g_model_image, self.model_image.trainable_variables))
			self.optimizer.apply_gradients(zip(self.g_model_state, self.model_state.trainable_variables))
			self.optimizer.apply_gradients(zip(self.g_model_inverse, self.model_inverse.trainable_variables))

			# if self.loss_lowpass is None: # TODO TEMP
			# 	self.loss_lowpass = self.loss_action_sum / self.loss_n
			# else:
			# 	self.loss_lowpass = 0.99*self.loss_lowpass + 0.01*(self.loss_action_sum / self.loss_n)

			print("epoch {:2d}: l_reward: {:8.5f}, l_action: {:8.5f} l_total: {:8.5f} l_lowpass: {:10.9f}".format(
				e, self.loss_reward_sum / self.loss_n,
				self.loss_action_sum / self.loss_n,
				self.loss_sum / self.loss_n,
				self.loss_lowpass))
			
			e += 1
		
			self.loss_action_avg = self.loss_action_sum / self.episode_length
	
	#@tf.function
	def train_image_autoencoder(self, image):
		n_epochs = 8
		#batch_size = 8

		g_model_image_encoder = None
		g_model_image_decoder = None

		n_entries = image.shape[0]

		image = tf.convert_to_tensor(image)
		for e in range(n_epochs):
			#image = tf.random.shuffle(image)
			loss_sum = 0.0
			for i in range(int(n_entries)):
				with tf.GradientTape(persistent=True) as gt:
					image_enc = self.model_image_encoder(image[i])
					image_pred = self.model_image_decoder(image_enc)
					loss = self.loss_image(image[i], image_pred)
					loss_sum += loss.numpy()
				
				g_model_image_encoder = gt.gradient(loss, self.model_image_encoder.trainable_variables)
				g_model_image_decoder = gt.gradient(loss, self.model_image_decoder.trainable_variables)
				
				print("{:2d} {:4d}/{:4d} ({})".format(e, i, n_entries, loss_sum/(i+1)), end="\r")
			
				self.optimizer.apply_gradients(zip(g_model_image_encoder,
					self.model_image_encoder.trainable_variables))
				self.optimizer.apply_gradients(zip(g_model_image_decoder,
					self.model_image_decoder.trainable_variables))
			

				if i % 4 == 0:
					img = cv2.hconcat([image[i,e].numpy(), image_pred[e].numpy()])
					cv2.imshow("target / prediction", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
					cv2.waitKey(10)


	def save_model(self, filename_prefix):
		self.model_image_encoder.save_weights("{}_image_encoder.h5".format(filename_prefix))
		self.model_image_decoder.save_weights("{}_image_decoder.h5".format(filename_prefix))
		#self.model_state.save("{}_state.h5".format(filename_prefix))
		#self.model_action.save("{}_action.h5".format(filename_prefix))
		#self.model_forward.save("{}_forward.h5".format(filename_prefix))
		#self.model_inverse.save("{}_inverse.h5".format(filename_prefix))
		#self.model_reward.save("{}_reward.h5".format(filename_prefix))
	
	def load_model(self, filename_prefix):
		self.model_image_encoder.load_weights("{}_image_encoder.h5".format(filename_prefix))
		self.model_image_decoder.load_weights("{}_image_decoder.h5".format(filename_prefix))
		#self.model_state = keras.models.load_model("{}_state.h5".format(filename_prefix))
		#self.model_action = keras.models.load_model("{}_action.h5".format(filename_prefix))
		#self.model_forward = keras.models.load_model("{}_forward.h5".format(filename_prefix))
		#self.model_inverse = keras.models.load_model("{}_inverse.h5".format(filename_prefix))
		#self.model_reward = keras.models.load_model("{}_reward.h5".format(filename_prefix))

		#self.create_recurrent_module()