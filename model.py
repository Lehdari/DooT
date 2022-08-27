from re import A
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras import regularizers
import random
from utils import *
from debug_tools import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import gc
import os
import cv2
from tensorflow.python.framework.ops import disable_eager_execution
from image_loss import ImageLoss
from git import Repo
import metrics

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class L2Regularizer(regularizers.Regularizer):
	def __init__(self, strength=1.0):
		self.strength = tf.Variable(strength)

	def __call__(self, x):
		return self.strength * tf.reduce_mean(tf.square(x))

class L8Regularizer(regularizers.Regularizer):
	def __init__(self, strength=1.0):
		self.strength = tf.Variable(strength)

	def __call__(self, x):
		return self.strength * tf.reduce_mean(tf.math.pow(x, 8.0))

class MaxRegularizer(regularizers.Regularizer):
	def __init__(self, strength=1.0, batch_size=8.0):
		self.strength = tf.Variable(strength)
		self.batch_size = tf.Variable(batch_size)

	def __call__(self, x):
		return self.strength * self.batch_size * tf.reduce_max(tf.abs(x))


def loss_function_inverse(action_true, action_pred):
	loss = tf.reduce_mean(tf.square(
		tf.math.sign(action_true[:,0:14,:])*0.5 - action_pred[:,0:14,:]))
	loss += tf.reduce_mean(tf.abs(action_true[:,14,:] - action_pred[:,14,:]))
	return loss


def learning_rate(t, t_max, t_min, lr_max, lr_min):
	if t <= t_max:
		return (t*(lr_max-lr_min))/t_max + lr_min
	else:
		return max(lr_max - ((t-t_max)*(lr_max-lr_min))/(t_min-t_max), lr_min)


class ActionModel:
	def __init__(self, model):
		self.model_state = model.model_state
		self.model_forward = model.model_forward
		self.model_reward = model.model_reward

		self.tbptt_length_action = model.tbptt_length_action
		self.action_optimizer = model.action_optimizer

		self.create_action_model(model)

		@tf.function(input_signature=[
			tf.TensorSpec(shape=(model.replay_sample_length, 8, model.image_enc_size), dtype=tf.float32),
			tf.TensorSpec(shape=(model.replay_sample_length, 8, 15), dtype=tf.float32),
			tf.TensorSpec(shape=(model.replay_sample_length, 8), dtype=tf.float32),
			tf.TensorSpec(shape=(8, model.state_size), dtype=tf.float32),
			tf.TensorSpec(shape=(), dtype=tf.int32),
			tf.TensorSpec(shape=(), dtype=tf.float32)
		])
		def train(image_encs, actions, rewards, state_init, i, discount_factor):
			discount_falloff = 1.0 # iterative discount factor
			discount_cum = 0.0
			state = self.model_state([state_init, image_encs[i]], training=False)
			with tf.GradientTape(persistent=True) as gt:
				action = self.model_action(state, training=True)
				reward = self.model_reward([state, action], training=True)
				image_enc = image_encs[i]

				reward_mean = tf.reduce_mean(reward)
				loss_reward = -reward_mean
				loss_reg = 2.0*tf.math.pow(self.model_action.losses[0], 4.0)*tf.abs(reward_mean)

				# simulate forward and predict rewards
				for j in range(1, self.tbptt_length_action):
					image_enc = self.model_forward([state, action], training=False)
					state = self.model_state([state, image_enc], training=False)
					action = self.model_action(state, training=True)
					reward = self.model_reward([state, action], training=True)

					reward_mean = tf.reduce_mean(reward)
					loss_reward -= reward_mean*discount_falloff
					loss_reg += 2.0*tf.math.pow(self.model_action.losses[0], 4.0)*\
						tf.abs(reward_mean)*discount_falloff
					
					discount_cum += discount_falloff
					discount_falloff *= discount_factor # update dc. falloff according to dc. factor
				
				loss_total = (loss_reward + loss_reg) / discount_cum
			
			g_model_action = gt.gradient(loss_total, self.model_action.trainable_variables)
			
			self.action_optimizer.apply_gradients(zip(g_model_action,
				self.model_action.trainable_variables))
			
			state = self.model_state([state_init, image_encs[i]], training=False)
			
			l_norm = 1.0 / self.tbptt_length_action
			return state, g_model_action, loss_total*l_norm, loss_reward*l_norm, loss_reg*l_norm
		
		self.train = train
	

	def create_action_model(self, model):
		self.model_action_i_state = keras.Input(shape=(model.state_size))

		x = model.module_dense(self.model_action_i_state, model.state_size, n2=model.state_size)

		self.model_action_o_action = layers.Dense(15,
			kernel_initializer=initializers.Orthogonal(), 
			activity_regularizer=MaxRegularizer(),
			use_bias=True, activation="tanh")(x)
		
		self.model_action = keras.Model(
			inputs=self.model_action_i_state,
			outputs=self.model_action_o_action,
			name="model_action")
		#self.model_action.summary()
	

	def save(self, filename):
		self.model_action.save_weights(filename)
	

	def load(self, filename):
		self.model_action.load_weights(filename)
	

	def __call__(self, input, training=True):
		return self.model_action(input, training=training)


class Model:
	def __init__(self,
		episode_length, n_replay_episodes, n_training_epochs, replay_sample_length,
		train_log_dir,
		output_visual_log=False,
		quiet=False,
		model_directory="models/model",
		learning_rate=0.001,
		momentum=0.9,
		architecture=1, # 2: wider, 3: deeper
		initializer=1, # 2: RandomNormal, 3: RandomUniform
		):
		self.model_directory = model_directory
		
		self.architecture = architecture # TODO temp, used only for grid search

		if initializer == 1:
			self.initializer = initializers.GlorotUniform()
		elif initializer == 2:
			self.initializer = initializers.RandomNormal(stddev=0.001)
		elif initializer == 3:
			self.initializer = initializers.RandomUniform(minval=-0.001, maxval=0.001)

		self.beta_initializer=initializers.RandomNormal(mean=0.0, stddev=0.0)
		self.gamma_initializer=initializers.RandomNormal(mean=1.0, stddev=0.0)
		self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=momentum, beta_2=0.999,
			clipnorm=0.1, clipvalue=1.0)
		self.action_optimizer = self.optimizer#keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
		self.loss_function = keras.losses.MeanSquaredError()

		self.state_size = 512
		self.image_enc_size = 4*4*128
		self.tbptt_length_encoder = 8
		self.tbptt_length_backbone = 64
		self.tbptt_length_action = 16

		self.reset_state()
		self.action_predict_step_size = tf.Variable(0.01)
		self.episode_length = episode_length
		self.n_replay_episodes = n_replay_episodes
		self.n_training_epochs = n_training_epochs
		self.replay_sample_length = replay_sample_length
		self.output_visual_log = output_visual_log

		self.create_image_encoder_model()
		self.create_image_decoder_model()

		# self.create_state_model()
		# self.create_reward_model()
		# self.create_forward_model()
		# self.create_inverse_models()
		# self.create_action_models()

		self.define_training_functions()

		self.image_flats = []
		self.image_encs = []
		self.image_enc_preds = []
		self.states = []

		np.random.seed(1507715517)

		self.train_log_dir = train_log_dir
		self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)

		self.num_epochs_init = 0
		self.quiet = quiet # suppress logging and rendering during training
		self.repo = Repo(".")

		
	
	def save_episode_state_images(self, episode_id):
		if len(self.image_encs) == 0:
			return
		
		cv2.imwrite("out/image_flat_{}.png".format(episode_id),
			(cv2.cvtColor(np.stack(self.image_flats, axis=1), cv2.COLOR_RGB2BGR)*65535).astype(np.uint16))
		cv2.imwrite("out/image_enc_{}.png".format(episode_id),
			(np.stack(self.image_encs, axis=-1)*65535).astype(np.uint16))
		cv2.imwrite("out/image_enc_pred_{}.png".format(episode_id),
			(np.stack(self.image_enc_preds, axis=-1)*65535).astype(np.uint16))
		cv2.imwrite("out/state_{}.png".format(episode_id),
			(np.stack(self.states, axis=-1)*65535).astype(np.uint16))

		self.image_flats = []
		self.image_encs = []
		self.image_enc_preds = []
		self.states = []


	def define_training_functions(self):

		@tf.function(input_signature=[
			tf.TensorSpec(shape=(self.replay_sample_length, 8, 240, 320, 3), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8, 240, 320, 1), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8, 15), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8), dtype=tf.float32),
			tf.TensorSpec(shape=(8, self.state_size), dtype=tf.float32),
			tf.TensorSpec(shape=(), dtype=tf.int32)
		])
		def train_image_encoder_model(image, automap, actions, rewards, state_init, i):
			with tf.GradientTape(persistent=True) as gt:
				image_enc = self.model_image_encoder(image, automap, training=True)
				state = self.model_state([state_init, image_enc], training=True)
				reward = self.model_reward([state, actions[i]], training=True)
				states = [state]

				loss_reward = self.loss_function(rewards[i], reward)
				loss_forward = tf.zeros_like(loss_reward)
				loss_reg = tf.reduce_mean(self.model_image_encoder.losses)

				for j in range(1, self.tbptt_length_encoder):
					image_enc_pred = self.model_forward([state, actions[i+j-1]])

					image_enc = self.model_image_encoder(image[i+j], automap[i+j], training=True)
					state = self.model_state([state, image_enc], training=True)
					reward = self.model_reward([state, actions[i+j]], training=True)
					states.append(state)
					
					# reward loss
					loss_reward += self.loss_function(rewards[i+j], reward)
					# forward loss
					loss_forward += tf.reduce_mean(tf.abs(image_enc - image_enc_pred))
					# regularization loss
					loss_reg += tf.reduce_mean(self.model_image_encoder.losses)

				actions_pred = self.model_inverse(tf.expand_dims(tf.stack(states, axis=-1), axis=-1),
					training=True)[:,:,:,0]
				actions_true = tf.transpose(actions[i:i+self.tbptt_length_encoder-1], perm=[1, 2, 0])
				
				loss_inverse = loss_function_inverse(actions_true, actions_pred)
				# loss_inverse *= 10.0 # TODO TEMP multiplier

				loss_forward /= (self.tbptt_length_encoder-1)

				loss_reward /= self.tbptt_length_encoder
				# loss_reward *= 0.1 # TODO TEMP multiplier

				loss_reg /= self.tbptt_length_encoder

				loss_total = loss_inverse + loss_reward + loss_forward + loss_reg

			
			g_model_image_encoder = gt.gradient(loss_total, self.model_image_encoder.trainable_variables)
			g_model_state = gt.gradient(loss_total, self.model_state.trainable_variables)
			g_model_inverse = gt.gradient(loss_total, self.model_inverse.trainable_variables)
			g_model_forward = gt.gradient(loss_total, self.model_forward.trainable_variables)
			g_model_reward = gt.gradient(loss_total, self.model_reward.trainable_variables)
			
			state = self.model_state([state_init, self.model_image_encoder(image[i], automap[i],
				training=False)], training=False)
			
			return state, g_model_image_encoder, g_model_state, g_model_inverse, g_model_forward, g_model_reward,\
				loss_total, loss_inverse, loss_forward, loss_reward


		@tf.function(input_signature=[
			tf.TensorSpec(shape=(self.replay_sample_length, 8, self.image_enc_size), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8, 15), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8), dtype=tf.float32),
			tf.TensorSpec(shape=(8, self.state_size), dtype=tf.float32),
			tf.TensorSpec(shape=(), dtype=tf.int32)
		])
		def train_backbone_inverse(image_encs, actions, rewards, state_init, i):
			with tf.GradientTape(persistent=True) as gt:
				state= self.model_state([state_init, image_encs[i]], training=True)
				reward = self.model_reward([state, actions[i]], training=True)
				states = [state]

				# loss_total = self.model_state.losses[0]
				loss_reward = self.loss_function(rewards[i], reward)
				loss_forward = tf.zeros_like(loss_reward)
				loss_reg = tf.reduce_mean(self.model_state.losses)

				for j in range(1, self.tbptt_length_backbone):
					image_enc_pred = self.model_forward([state, actions[i+j-1]])

					state = self.model_state([state, image_encs[i+j]], training=True)
					reward = self.model_reward([state, actions[i+j]], training=True)
					states.append(state)
					
					# reward loss
					loss_reward += self.loss_function(rewards[i+j], reward)
					# forward loss
					loss_forward += tf.reduce_mean(tf.abs(image_encs[i+j] - image_enc_pred))
					loss_reg += tf.reduce_mean(self.model_state.losses)

				actions_pred = self.model_inverse_backbone(
					tf.expand_dims(tf.stack(states, axis=-1), axis=-1), training=True)[:,:,:,0]
				actions_true = tf.transpose(actions[i:i+self.tbptt_length_backbone-1], perm=[1, 2, 0])
				
				loss_inverse = loss_function_inverse(actions_true, actions_pred)
				# loss_inverse *= 10.0 # TODO TEMP multiplier

				loss_forward /= (self.tbptt_length_backbone-1)

				loss_reward /= self.tbptt_length_backbone
				# loss_reward *= 0.1 # TODO TEMP multiplier

				loss_reg /= self.tbptt_length_backbone

				loss_total = loss_inverse + loss_reward + loss_forward + loss_reg
			
			g_model_state = gt.gradient(loss_total, self.model_state.trainable_variables)
			g_model_inverse_backbone = gt.gradient(loss_total,
				self.model_inverse_backbone.trainable_variables)
			g_model_forward = gt.gradient(loss_total, self.model_forward.trainable_variables)
			g_model_reward = gt.gradient(loss_total, self.model_reward.trainable_variables)
			
			state = self.model_state([state_init, image_encs[i]], training=False)
			
			# l_norm = 1.0 / self.tbptt_length_encoder
			return state, g_model_state, g_model_inverse_backbone, g_model_forward, g_model_reward,\
				loss_total, loss_inverse, loss_forward, loss_reward

		@tf.function(input_signature=[
			tf.TensorSpec(shape=(self.replay_sample_length, 8, self.image_enc_size), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8, 15), dtype=tf.float32),
			tf.TensorSpec(shape=(self.replay_sample_length, 8), dtype=tf.float32),
			tf.TensorSpec(shape=(8, self.state_size), dtype=tf.float32),
			tf.TensorSpec(shape=(), dtype=tf.int32)
		])
		def train_backbone(image_encs, actions, rewards, state_init, i):
			discount_factor = 1.0
			discount_cum = 0.0
			with tf.GradientTape(persistent=True) as gt:
				state = self.model_state([state_init, image_encs[i]], training=True)
				# reward = self.model_reward([state, actions[i]], training=True)
				image_enc = image_encs[i]

				# loss_total = self.model_image_encoder.losses[0] + self.model_state.losses[0]
				# loss_reward = self.loss_function(rewards[i], reward)
				loss_forward = tf.zeros(shape=())

				for j in range(1, self.tbptt_length_backbone):
					image_enc = self.model_forward([state, actions[i+j-1]], training=True)
					state = self.model_state([state, image_enc], training=True)
					# reward = self.model_reward([state, actions[i+j]], training=True)

					loss_enc_iter = tf.reduce_mean(tf.abs(image_encs[i+j] - image_enc))
					loss_forward += loss_enc_iter * discount_factor
					# loss_reward += self.loss_function(rewards[i+j], reward) * discount_factor
					# loss_total += self.model_image_encoder.losses[0] + self.model_state.losses[0]

					# discount falloff according to prediction error
					discount_cum += discount_factor
					discount_factor *= tf.clip_by_value(1.0-loss_enc_iter, 1.0e-10, 1.0)
				
				# loss_forward *= 10.0 # TODO TEMP?
				# loss_reward /= discount_cum # normalize by cumulative discount
				loss_forward /= discount_cum
				# loss_total += loss_forward#+ loss_reward
			
			g_model_forward = gt.gradient(loss_forward, self.model_forward.trainable_variables)
			# g_model_state = gt.gradient(loss_total, self.model_state.trainable_variables)
			# g_model_reward = gt.gradient(loss_total, self.model_reward.trainable_variables)
		
			# self.optimizer.apply_gradients(zip(g_model_forward,
			# 	self.model_forward.trainable_variables))
			# self.optimizer.apply_gradients(zip(g_model_state,
			# 	self.model_state.trainable_variables))
			# self.optimizer.apply_gradients(zip(g_model_reward,
			# 	self.model_reward.trainable_variables))
			
			state = self.model_state([state_init, image_encs[i]], training=False)
			return state, g_model_forward, loss_forward, discount_cum
		
		@tf.function(input_signature=[
			tf.TensorSpec(shape=(self.replay_sample_length, 8, 240, 320, 5), dtype=tf.float32),
			tf.TensorSpec(shape=(), dtype=tf.int32)
		])
		def train_autoencoder(images, i):
			with tf.GradientTape(persistent=True) as gt:
				image = images[i][:,:,:,1:4]
				automap = images[i][:,:,:,0:1]
				image_enc = self.model_image_encoder([image, automap], training=True)
				image_pred, depth_pred = self.model_image_decoder(image_enc, training=True)

				image_combined = images[i][:,:,:,1:]
				image_combined_pred = layers.Concatenate()([image_pred, depth_pred])
				
				image_loss = ImageLoss(image_combined, image_combined_pred)
				loss_regularizer = self.model_image_encoder.losses
				loss_regularizer += self.model_image_decoder.losses
				loss_total = image_loss.total_loss() + loss_regularizer

			g_model_image_encoder = gt.gradient(loss_total, self.model_image_encoder.trainable_variables)
			g_model_image_decoder = gt.gradient(loss_total, self.model_image_decoder.trainable_variables)

			return image_pred, depth_pred, g_model_image_encoder, g_model_image_decoder,\
				loss_total, loss_regularizer

		self.train_image_encoder_model = train_image_encoder_model
		self.train_backbone_inverse = train_backbone_inverse
		self.train_backbone = train_backbone
		self.train_autoencoder = train_autoencoder

	
	def module_dense(self, x, n, x2=None, n2=None, alpha=0.001, act=None):
		use_shortcut = n == x.shape[1]

		if use_shortcut:
			y = x

		# enable concatenation if auxiliary input is provided
		if x2 is not None:
			x = layers.Concatenate()([x, x2])
		
		# double layer model
		if n2 is not None:
			x = layers.Dense(n2, kernel_initializer=self.initializer, use_bias=False)(x)
			x = layers.BatchNormalization(axis=-1,
				beta_initializer = self.beta_initializer,
				gamma_initializer = self.gamma_initializer)(x)
			x = activations.relu(x, alpha=alpha)

		x = layers.Dense(n, kernel_initializer=self.initializer, use_bias=False)(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer = self.beta_initializer,
			gamma_initializer = self.gamma_initializer)(x)
		x = activations.relu(x, alpha=alpha)

		if use_shortcut:
			x = layers.Add()([x, y])
		
		if act is not None:
			x = act(x)
		
		return x
	

	def module_dense2(self, x, n, alpha=0.001):
		use_shortcut = n == x.shape[1]

		if use_shortcut:
			y = x

		x = layers.Dense(n, use_bias=False,
			kernel_initializer=initializers.Orthogonal())(x)
		x = layers.BatchNormalization(axis=-1,
			beta_initializer = self.beta_initializer,
			gamma_initializer = self.gamma_initializer)(x)
		x = activations.relu(x, alpha=alpha)

		x = layers.Dense(n, use_bias=True,
			kernel_initializer=initializers.Orthogonal())(x)
		
		if use_shortcut:
			x = layers.Add()([x, y])

		return x


	def module_conv(self, x, n1, n2, k1=(3,3), k2=(3,3), s1=(2,2), s2=(1,1),
		p1="same", p2="same", activation1="relu", activation2="relu", alpha=0.001,
		use_post_activation=False,
		initializer_primary=initializers.Orthogonal(0.5),
		initializer_secondary=initializers.Orthogonal()):

		#shortcut by avg pooling
		pool_x_size = s1[0]*s2[0]
		pool_y_size = s1[1]*s2[1]
		skip_x_kernel = 1
		skip_y_kernel = 1
		if p1 == "valid":
			skip_x_kernel += k1[0]-1
			skip_y_kernel += k1[1]-1
		if p2 == "valid":
			skip_x_kernel += k2[0]-1
			skip_y_kernel += k2[1]-1
		
		y = x
		if pool_x_size != 1 or pool_y_size != 1:
			y = layers.AveragePooling2D((pool_x_size, pool_y_size))(y)	
		if n2 != x.shape[3] or skip_x_kernel > 1 or skip_y_kernel > 1:
			y = layers.BatchNormalization(axis=-1,
				beta_initializer = self.beta_initializer,
				gamma_initializer = self.gamma_initializer)(y)
			y = layers.Conv2D(n2, (skip_x_kernel, skip_y_kernel),
				kernel_initializer=initializer_primary, use_bias=False, padding="valid")(y)

		x = layers.BatchNormalization(axis=-1,
			beta_initializer = self.beta_initializer,
			gamma_initializer = self.gamma_initializer)(x)
		x = layers.Conv2D(n1, k1, padding=p1, kernel_initializer=initializer_secondary,
			strides=s1, use_bias=False)(x)
		if activation1 == "tanh":
			x = layers.Activation(activations.tanh)(x)
		if activation1 == "sigmoid":
			x = layers.Activation(activations.sigmoid)(x)
		elif activation1 == "relu":
			x = activations.relu(x, alpha=alpha)
		else:
			pass

		x = layers.BatchNormalization(axis=-1,
			beta_initializer = self.beta_initializer,
			gamma_initializer = self.gamma_initializer)(x)
		x = layers.Conv2D(n2, k2, padding=p2, kernel_initializer=initializer_primary,
			strides=s2, use_bias=False)(x)
		if not use_post_activation:
			if activation2 == "tanh":
				x = layers.Activation(activations.tanh)(x)
			if activation2 == "sigmoid":
				x = layers.Activation(activations.sigmoid)(x)
			elif activation2 == "relu":
				x = activations.relu(x, alpha=alpha)
			else:
				pass

		x = layers.Add()([x, y])
		if use_post_activation:
			if activation2 == "tanh":
				x = layers.Activation(activations.tanh)(x)
			if activation2 == "sigmoid":
				x = layers.Activation(activations.sigmoid)(x)
			elif activation2 == "relu":
				x = activations.relu(x, alpha=alpha)
			else:
				pass
		
		return x
	

	def module_deconv(self, x, n1, n2, k1=(4,4), k2=(2,2), s1=(2,2), s2=(1,1),
		p1="same", p2="same", activation1="relu", activation2="relu", alpha=0.001,
		use_post_activation=False,
		initializer_primary=initializers.Orthogonal(0.5),
		initializer_secondary=initializers.Orthogonal()):

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
		if n2 != x.shape[3]:
			y = layers.BatchNormalization(axis=-1,
				beta_initializer = self.beta_initializer,
				gamma_initializer = self.gamma_initializer)(y)
			y = layers.Conv2D(n2, (1, 1), kernel_initializer=initializer_primary, use_bias=False)(y)

		x = layers.BatchNormalization(axis=-1,
			beta_initializer = self.beta_initializer,
			gamma_initializer = self.gamma_initializer)(x)
		x = layers.Conv2DTranspose(n1, k1, padding=p1, kernel_initializer=initializer_secondary,
			strides=s1, use_bias=False)(x)
		if activation1 == "tanh":
			x = layers.Activation(activations.tanh)(x)
		if activation1 == "sigmoid":
			x = layers.Activation(activations.sigmoid)(x)
		elif activation1 == "relu":
			x = activations.relu(x, alpha=alpha)
		else:
			pass

		x = layers.BatchNormalization(axis=-1,
			beta_initializer = self.beta_initializer,
			gamma_initializer = self.gamma_initializer)(x)
		x = layers.Conv2DTranspose(n2, k2, padding=p2, kernel_initializer=initializer_primary,
			strides=s2, use_bias=False)(x)
		if not use_post_activation:
			if activation2 == "tanh":
				x = layers.Activation(activations.tanh)(x)
			if activation2 == "sigmoid":
				x = layers.Activation(activations.sigmoid)(x)
			elif activation2 == "relu":
				x = activations.relu(x, alpha=alpha)
			else:
				pass

		x = layers.Add()([x, y])
		if use_post_activation:
			if activation2 == "tanh":
				x = layers.Activation(activations.tanh)(x)
			if activation2 == "sigmoid":
				x = layers.Activation(activations.sigmoid)(x)
			elif activation2 == "relu":
				x = activations.relu(x, alpha=alpha)
			else:
				pass

		return x

	
	def module_deconv_no_upsampling(self, x, n1, n2, k=(2,2), s=(1,1),
		p="same", activation1="relu", activation2="relu", alpha=0.001,
		use_post_activation=False,
		initializer_primary=initializers.Orthogonal(0.5),
		initializer_secondary=initializers.Orthogonal()):

		#shortcut by linear upsampling
		y = layers.BatchNormalization(axis=-1,
			beta_initializer = self.beta_initializer,
			gamma_initializer = self.gamma_initializer)(x)
		y = layers.Conv2DTranspose(n2, k, padding=p, strides=s,
			kernel_initializer=initializer_primary, use_bias=False)(y)

		x = layers.BatchNormalization(axis=-1,
			beta_initializer = self.beta_initializer,
			gamma_initializer = self.gamma_initializer)(x)
		x = layers.Conv2DTranspose(n1, k, padding=p, kernel_initializer=initializer_secondary,
			strides=s, use_bias=False)(x)
		if activation1 == "tanh":
			x = layers.Activation(activations.tanh)(x)
		if activation1 == "sigmoid":
			x = layers.Activation(activations.sigmoid)(x)
		elif activation1 == "relu":
			x = activations.relu(x, alpha=alpha)
		else:
			pass

		x = layers.BatchNormalization(axis=-1,
			beta_initializer = self.beta_initializer,
			gamma_initializer = self.gamma_initializer)(x)
		x = layers.Conv2D(n2, (1,1), kernel_initializer=initializer_primary,
			strides=s, use_bias=False)(x)
		if not use_post_activation:
			if activation2 == "tanh":
				x = layers.Activation(activations.tanh)(x)
			if activation2 == "sigmoid":
				x = layers.Activation(activations.sigmoid)(x)
			elif activation2 == "relu":
				x = activations.relu(x, alpha=alpha)
			else:
				pass

		x = layers.Add()([x, y])
		if use_post_activation:
			if activation2 == "tanh":
				x = layers.Activation(activations.tanh)(x)
			if activation2 == "sigmoid":
				x = layers.Activation(activations.sigmoid)(x)
			elif activation2 == "relu":
				x = activations.relu(x, alpha=alpha)
			else:
				pass

		return x
	

	def module_inception(self, x, n=[], k=[], alpha=0.001):
		n_total = sum(n)

		y = x
		if n_total != x.shape[3]:
			y = layers.Conv2D(n_total, (1, 1),
				kernel_initializer=self.initializer, use_bias=False)(y)
			y = layers.BatchNormalization(axis=-1,
				beta_initializer = self.beta_initializer,
				gamma_initializer = self.gamma_initializer)(y)

		xv = []
		for i in range(len(n)):
			xv.append(layers.Conv2D(n[i], k[i], padding="same",
				kernel_initializer=self.initializer, use_bias=False)(x))
			xv[-1] = layers.BatchNormalization(axis=-1,
				beta_initializer = self.beta_initializer,
				gamma_initializer = self.gamma_initializer)(xv[-1])
			xv[-1] = activations.relu(xv[-1], alpha=alpha)

		x = layers.concatenate(xv)

		return layers.Add()([x, y])

	
	def module_fusion(self, x1, x2, n, activation="tanh",
		initializer=initializers.Orthogonal(1.0), alpha=0.001):
		# gates
		g1 = layers.Concatenate()([x1, x2])
		g1 = layers.BatchNormalization(axis=-1,
			beta_initializer = self.beta_initializer,
			gamma_initializer = self.gamma_initializer)(g1)
		g1 = layers.Dense(n,
			kernel_initializer=initializer,
			use_bias=False, activation="sigmoid")(g1)
		g2 = layers.Lambda(lambda x: 1.0 - x)(g1)

		# adapter dense layers
		x1 = layers.BatchNormalization(axis=-1,
			beta_initializer = self.beta_initializer,
			gamma_initializer = self.gamma_initializer)(x1)
		x1 = layers.Dense(n, kernel_initializer=initializer, use_bias=False)(x1)
		if activation == "tanh":
			x1 = layers.Activation(activations.tanh)(x1)
		elif activation == "relu":
			x1 = activations.relu(x1, alpha=alpha)
		elif activation == "linear":
			pass

		x2 = layers.BatchNormalization(axis=-1,
			beta_initializer = self.beta_initializer,
			gamma_initializer = self.gamma_initializer)(x2)
		x2 = layers.Dense(n, kernel_initializer=initializer, use_bias=False)(x2)
		if activation == "tanh":
			x2 = layers.Activation(activations.tanh)(x2)
		elif activation == "relu":
			x2 = activations.relu(x2, alpha=alpha)
		elif activation == "linear":
			pass
 
		return layers.Add()([layers.Multiply()([x1, g1]), layers.Multiply()([x2, g2])])

	
	def module_projected_sin(self, x, n=1, s=(320, 240), scale=2.0/320):
		#output shape: (s(0), s(1), n*2)

		batch_size = tf.shape(x)[0]

		x = layers.Flatten()(x)
		x = layers.Dense(8*n,
			kernel_initializer=initializers.Orthogonal(0.001),
			activity_regularizer=L2Regularizer(0.01))(x)
		x = layers.Reshape((n,8))(x)
		x = tf.concat([x, tf.ones((batch_size, n, 1))], axis=2)
		x = tf.reshape(x, [batch_size*n,3,3])

		i = tf.repeat(tf.expand_dims(tf.eye(3), axis=0), batch_size*n, axis=0)
		xx, yy = tf.meshgrid(
			tf.linspace(-s[0]+0.5, s[0]-0.5, s[0]),
			tf.linspace(-s[1]+0.5, s[1]-0.5, s[1])
		)

		xx = tf.expand_dims(xx, axis=0) * scale
		yy = tf.expand_dims(yy, axis=0) * scale
		p = tf.concat([xx, yy, tf.ones_like(xx)], axis=0)
		p = tf.reshape(p, [3, s[0]*s[1]])

		# projection
		x = tf.linalg.matmul(i+x, tf.broadcast_to(p, [batch_size*n, *p.shape]))
		
		x = tf.reshape(x, [batch_size*n, 3, s[1], s[0]])

		# depth division
		x2 = x[:,:2,:,:] / (x[:,2:3,:,:] + 1.0e-5)

		x2 = tf.reshape(x2, [batch_size, 2*n, s[1], s[0]])
		x2 = tf.transpose(x2, perm=[0,2,3,1])

		x2 = tf.math.sin(x2)

		return x2

	
	def module_interpolated_sin(self, x, n=1, upscale_ratio=2):
		batch_size = tf.shape(x)[0]
		in_rows = x.get_shape().as_list()[1]
		in_cols = x.get_shape().as_list()[2]
		
		s1 = int(np.sqrt(upscale_ratio))
		s2 = int(upscale_ratio/s1) # TODO error in case the result is not an integer

		x = layers.Conv2D(8*n, (1,1),
			kernel_initializer=initializers.Orthogonal(1.0))(x)
		x = layers.UpSampling2D(size=(upscale_ratio, upscale_ratio), interpolation="bilinear")(x)
		x = layers.Activation(activations.tanh)(x)

		# translation coefficients
		t1 = layers.Conv2D(2*n, (1,1), padding="same",
			kernel_initializer=initializers.Orthogonal(1.0))(x)
		t2 = layers.Conv2D(2*n, (1,1), padding="same",
			kernel_initializer=initializers.Orthogonal(1.0))(x)
		# transformation matrix coefficients
		x = layers.Conv2D(4*n, (1,1), padding="same",
			kernel_initializer=initializers.RandomNormal(stddev=1.0))(x)

		out_rows = x.get_shape().as_list()[1]
		out_cols = x.get_shape().as_list()[2]
		matmul_batch_shape = (batch_size, out_rows, out_cols, n)

		# reshape coefficients into matrices
		x = tf.reshape(x, [*matmul_batch_shape, 2, 2])

		# transform and final reshape
		t1 = tf.reshape(t1, [*matmul_batch_shape, 2])
		x = tf.linalg.matvec(x, t1)
		x = tf.reshape(x, [batch_size, out_rows, out_cols, 2*n])

		# translation and sine activation
		x = tf.math.sin(x + t2)
		return x


	@staticmethod
	def module_positional_embedding(x, n):
		shape = x.get_shape().as_list()

		x_mesh, y_mesh = tf.meshgrid(
			tf.linspace(-shape[2]//2+0.5, shape[2]//2-0.5, shape[2]),
			tf.linspace(-shape[1]//2+0.5, shape[1]//2-0.5, shape[1])
		)

		emb_list = []
		for i in range(n):
			emb_list.append(tf.sin((x_mesh*np.pi)/(2**i)))
			emb_list.append(tf.sin((y_mesh*np.pi)/(2**i)))

		e = tf.broadcast_to(tf.expand_dims(tf.constant(tf.stack(emb_list, axis=2)), 0),
			[tf.shape(x)[0], shape[1], shape[2], n*2])
		
		return layers.Concatenate()([x, e])


	def create_positional_embedding_mesh(self, shape):
		x = tf.convert_to_tensor([
			[-1.0, -1.0],
			[1.0, 1.0]])
		y = tf.convert_to_tensor([
			[-1.0, 1.0],
			[-1.0, 1.0]])
		xy = tf.stack([x, y], axis=2)

		xyr = tf.tile(xy, multiples=[int(shape[0]/2), int(shape[1]/2), 1])

		return tf.constant(xyr)


	def create_image_encoder_model(self):
		self.model_image_encoder_i_image = keras.Input(shape=(240, 320, 3))
		self.model_image_encoder_i_automap = keras.Input(shape=(240, 320, 1))

		if self.architecture == 1:
			x = layers.Concatenate()([
				self.model_image_encoder_i_image, self.model_image_encoder_i_automap])
			x = self.module_conv(x, 12, 16, k1=(3,2), s1=(3,2), k2=(2,4), s2=(1,2),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 80x80
			x = self.module_conv(x, 16, 16, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 40x40
			x = self.module_conv(x, 32, 32, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 20x20
			x = self.module_conv(x, 64, 64, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 10x10
			x = self.module_conv(x, 128, 128, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 5x5
			x = self.module_conv(x, 128, 128, k1=(2,2), s1=(1,1), k2=(1,1), s2=(1,1),
				p1="valid", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 4x4
			x = layers.Conv2D(128, (1,1),
				kernel_initializer=self.initializer,
				activity_regularizer=L2Regularizer(1.0e-5))(x)
			self.model_image_encoder_o_image_enc = layers.Flatten()(x)

		elif self.architecture == 2: # the wider architecture
			x = layers.Concatenate()([
				self.model_image_encoder_i_image, self.model_image_encoder_i_automap])
			x = self.module_conv(x, 18, 24, k1=(3,2), s1=(3,2), k2=(2,4), s2=(1,2),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 80x80
			x = self.module_conv(x, 24, 24, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 40x40
			x = self.module_conv(x, 48, 48, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 20x20
			x = self.module_conv(x, 96, 96, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 10x10
			x = self.module_conv(x, 192, 192, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 5x5
			x = self.module_conv(x, 192, 192, k1=(2,2), s1=(1,1), k2=(1,1), s2=(1,1),
				p1="valid", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 4x4
			x = layers.Conv2D(128, (1,1),
				kernel_initializer=self.initializer,
				activity_regularizer=L2Regularizer(1.0e-5))(x)
			self.model_image_encoder_o_image_enc = layers.Flatten()(x)

		else: # the split architecture
			x = self.module_conv(self.model_image_encoder_i_image,
				12, 16, k1=(3,2), s1=(3,2), k2=(2,4), s2=(1,2),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 80x80
			x = self.module_conv(x, 16, 16, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 40x40
			x = self.module_conv(x, 32, 32, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 20x20
			x = self.module_conv(x, 64, 64, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 10x10
			x = self.module_conv(x, 128, 128, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 5x5
			x = self.module_conv(x, 128, 128, k1=(2,2), s1=(1,1), k2=(1,1), s2=(1,1),
				p1="valid", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 4x4
			x = layers.Conv2D(96, (1,1),
				kernel_initializer=self.initializer,
				activity_regularizer=L2Regularizer(1.0e-5))(x)
			x = layers.Flatten()(x)

			y = self.module_conv(self.model_image_encoder_i_automap,
				6, 8, k1=(3,2), s1=(3,2), k2=(2,4), s2=(1,2),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 80x80
			y = self.module_conv(y, 8, 8, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 40x40
			y = self.module_conv(y, 16, 16, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 20x20
			y = self.module_conv(y, 32, 32, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 10x10
			y = self.module_conv(y, 64, 64, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
				p1="same", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 5x5
			y = self.module_conv(y, 64, 64, k1=(2,2), s1=(1,1), k2=(1,1), s2=(1,1),
				p1="valid", p2="same", activation1="relu", activation2="tanh",
				initializer_primary=self.initializer,
				initializer_secondary=self.initializer) # 4x4
			y = layers.Conv2D(32, (1,1),
				kernel_initializer=self.initializer,
				activity_regularizer=L2Regularizer(1.0e-5))(y)
			y = layers.Flatten()(y)

			y = self.module_fusion(x, y, 512, initializer=self.initializer)
			self.model_image_encoder_o_image_enc = layers.Concatenate()([x, y])

		self.model_image_encoder = keras.Model(
			inputs=[
				self.model_image_encoder_i_image,
				self.model_image_encoder_i_automap
			],
			outputs=self.model_image_encoder_o_image_enc,
			name="model_image_encoder")
		# self.model_image_encoder.summary()


	def create_image_decoder_model(self):
		self.model_image_decoder_i_image_enc = keras.Input(shape=(self.image_enc_size))

		x = self.model_image_decoder_i_image_enc
		if self.architecture == 3:
			# extract and redistribute both encoding split sections separately
			x = layers.Reshape((4, 4, 32))(x[:,1536:2048]) 

			y = self.model_image_decoder_i_image_enc
			y = layers.Reshape((4, 4, 96))(y[:,0:1536]) 

			x = layers.Concatenate()([x, y])

		else:
			x = layers.Reshape((4, 4, -1))(x)
		
		x = self.module_deconv_no_upsampling(x, 512, 256, k=(2,2),
			p="valid", activation1="tanh", activation2="tanh", use_post_activation=True,
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer) # 5x5
		x = self.module_conv(x, 256, 256, k1=(3,1), s1=(1,1), k2=(1,3), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer)

		x = self.module_deconv(x, 512, 256, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer) # 10x10
		x = self.module_conv(x, 256, 256, k1=(3,1), s1=(1,1), k2=(1,3), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer)
		x = self.module_conv(x, 256, 256, k1=(3,1), s1=(1,1), k2=(1,3), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer)
		
		x = self.module_deconv(x, 256, 128, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer) # 20x20
		x = self.module_conv(x, 128, 128, k1=(3,1), s1=(1,1), k2=(1,3), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer)
		x = self.module_conv(x, 128, 128, k1=(3,1), s1=(1,1), k2=(1,3), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer)
		
		x = self.module_deconv(x, 128, 64, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer) # 40x40
		x = self.module_conv(x, 64, 64, k1=(3,1), s1=(1,1), k2=(1,3), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer)
		x = self.module_conv(x, 64, 64, k1=(3,1), s1=(1,1), k2=(1,3), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer)

		x = self.module_deconv(x, 64, 32, k1=(4,4), s1=(2,2), k2=(2,2), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer) # 80x80
		x = self.module_conv(x, 32, 32, k1=(3,1), s1=(1,1), k2=(1,3), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer)
		x = self.module_conv(x, 32, 32, k1=(3,1), s1=(1,1), k2=(1,3), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer)

		x = self.module_deconv(x, 32, 32, k1=(6,4), s1=(3,2), k2=(2,4), s2=(1,2),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer) # 240x320
		x = self.module_conv(x, 32, 32, k1=(3,1), s1=(1,1), k2=(1,3), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer)
		x = self.module_conv(x, 32, 32, k1=(3,1), s1=(1,1), k2=(1,3), s2=(1,1),
			activation1="tanh", activation2="tanh",
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer)

		y = self.module_conv(x, 8, 1, k1=(3,3), s1=(1,1), k2=(1,1), s2=(1,1),
			activation1="tanh", activation2="sigmoid", use_post_activation=True,
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer) # luminosity

		z = self.module_conv(x, 8, 2, k1=(3,3), s1=(1,1), k2=(1,1), s2=(1,1),
			activation1="tanh", activation2="sigmoid", use_post_activation=True,
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer) - 0.5 # chromaticity

		self.model_image_decoder_o_image = layers.Concatenate()([y, z])
		self.model_image_decoder_o_depth = self.module_conv(x, 8, 1,
			k1=(3,3), s1=(1,1), k2=(1,1), s2=(1,1),
			activation1="tanh", activation2="linear", use_post_activation=True,
			initializer_primary=self.initializer,
			initializer_secondary=self.initializer)

		self.model_image_decoder = keras.Model(
			inputs=[self.model_image_decoder_i_image_enc],
			outputs=[
				self.model_image_decoder_o_image,
				self.model_image_decoder_o_depth],
			name="model_image_decoder")
		# self.model_image_decoder.summary()


	def create_state_model(self):
		self.model_state_i_state = keras.Input(shape=(self.state_size))
		self.model_state_i_image_enc = keras.Input(shape=(self.image_enc_size))

		# reset gate
		r_i = self.module_dense2(self.model_state_i_image_enc, self.state_size)
		r_s = self.module_dense2(self.model_state_i_state, self.state_size)
		r = layers.Add()([r_i, r_s])
		r = layers.Dense(self.state_size, use_bias=True,
			bias_initializer=initializers.RandomUniform(-0.5, 1.5),
			kernel_initializer=initializers.Orthogonal(),
			activation="sigmoid")(r)

		# update gate
		u_i = self.module_dense2(self.model_state_i_image_enc, self.state_size)
		u_s = self.module_dense2(self.model_state_i_state, self.state_size)
		u1 = layers.Add()([u_i, u_s])
		u1 = layers.Dense(self.state_size, use_bias=True,
			bias_initializer=initializers.RandomUniform(-1.0, 4.0),
			kernel_initializer=initializers.Orthogonal(),
			activation="sigmoid")(u1)
		u2 = layers.Lambda(lambda x: 1.0 - x)(u1)

		# image encoding / action fusion
		i_i = layers.Dense(self.state_size, kernel_initializer=initializers.Orthogonal(),
			use_bias=True)(self.model_state_i_image_enc)
		i_i = layers.BatchNormalization(axis=-1,
			beta_initializer = self.beta_initializer,
			gamma_initializer = self.gamma_initializer)(i_i)
		i_i = layers.Activation(activations.tanh)(i_i)

		# candidate state
		s = layers.Add()([layers.Multiply()([self.model_state_i_state, r]), i_i])
		s = self.module_dense(s, self.state_size, n2=self.state_size)
		s = layers.Dense(self.state_size, use_bias=True,
			kernel_initializer=initializers.Orthogonal(),
			activation="tanh",
			activity_regularizer=L8Regularizer(1.0))(s)

		# output state
		self.model_state_o_state = layers.Add()([
			layers.Multiply()([self.model_state_i_state, u1]), layers.Multiply()([s, u2]) ])

		self.model_state = keras.Model(
			inputs=[self.model_state_i_state, self.model_state_i_image_enc],
			outputs=self.model_state_o_state,
			name="model_state")
		# self.model_state.summary()


	def create_reward_model(self):
		self.model_reward_i_state = keras.Input(shape=(self.state_size))
		self.model_reward_i_action = keras.Input(shape=(15))

		x = self.module_dense(self.model_reward_i_state, self.state_size,
			x2=self.model_reward_i_action, n2=self.state_size)
		x = self.module_dense(x, self.state_size, n2=self.state_size)
		x = self.module_dense(x, self.state_size, n2=self.state_size)

		self.model_reward_o_reward_step = layers.Dense(1,
			kernel_initializer=self.initializer,
			use_bias=True, bias_initializer=self.initializer)(x)

		self.model_reward = keras.Model(
			inputs=[self.model_reward_i_state, self.model_reward_i_action],
			outputs=[self.model_reward_o_reward_step],
			name="model_reward")
		# self.model_reward.summary()
	

	# predict encoding of next image from state and action
	def create_forward_model(self):
		self.model_forward_i_state = keras.Input(shape=(self.state_size))
		self.model_forward_i_action = keras.Input(shape=(15))

		x = self.module_fusion(self.model_forward_i_state, self.model_forward_i_action,
			self.image_enc_size, activation="relu")
		x = self.module_dense(x, self.state_size, n2=self.state_size)
		x = self.module_dense(x, self.image_enc_size, n2=self.image_enc_size)

		self.model_forward_o_image_enc = layers.Dense(self.image_enc_size,
			kernel_initializer=self.initializer, activation="tanh")(x)

		self.model_forward = keras.Model(
			inputs=[self.model_forward_i_state, self.model_forward_i_action],
			outputs=[self.model_forward_o_image_enc],
			name="model_forward")
		# self.model_forward.summary()


	def create_inverse_models(self):
		l_e = self.tbptt_length_encoder # alias for easier usage
		s8 = int(self.state_size/8)
		alpha = 0.001

		comp_dense1 = layers.Dense(self.state_size,
			kernel_initializer=initializers.Orthogonal(), use_bias=False)
		comp_bn_dense1 = layers.BatchNormalization(axis=-1,
			beta_initializer = self.beta_initializer,
			gamma_initializer = self.gamma_initializer)
		
		comp_dense2 = layers.Dense(15,
			kernel_initializer=initializers.Orthogonal(), use_bias=True)


		self.model_inverse_i_states = keras.Input(shape=(self.state_size, l_e, 1))

		# process all states with the shared layers into a single block
		compressed = []
		for i in range(l_e-1):
			x = layers.Flatten()(self.model_inverse_i_states[:, :, i:i+2, :])
			x = comp_bn_dense1(comp_dense1(x))
			x = activations.relu(x, alpha=alpha)
			x = comp_dense2(x)
			compressed.append(x)
		x = tf.expand_dims(tf.stack(compressed, axis=2), axis=-1)

		self.model_inverse_o_actions = layers.Activation(activations.tanh)(x)

		self.model_inverse = keras.Model(
			inputs=self.model_inverse_i_states,
			outputs=self.model_inverse_o_actions,
			name="model_inverse")
		# self.model_inverse.summary()
	

		l_e = self.tbptt_length_backbone # alias for easier usage

		self.model_inverse_backbone_i_states = keras.Input(shape=(self.state_size, l_e, 1))

		compressed = []
		for i in range(l_e-1):
			x = layers.Flatten()(self.model_inverse_backbone_i_states[:, :, i:i+2, :])
			x = comp_bn_dense1(comp_dense1(x))
			x = activations.relu(x, alpha=alpha)
			x = comp_dense2(x)
			compressed.append(x)
		x = tf.expand_dims(tf.stack(compressed, axis=2), axis=-1)
		
		self.model_inverse_backbone_o_actions = layers.Activation(activations.tanh)(x)

		self.model_inverse_backbone = keras.Model(
			inputs=self.model_inverse_backbone_i_states,
			outputs=self.model_inverse_backbone_o_actions,
			name="model_inverse_backbone")
		# self.model_inverse_backbone.summary()


	def create_action_models(self):
		self.models_action = []
		for i in range(self.n_replay_episodes):
			self.models_action.append(ActionModel(self))


	def advance(self, image, action_prev):
		# preprocess image and previous action
		image = tf.convert_to_tensor(image, dtype=tf.float32) * 0.0039215686274509803 # 1/255
		action_prev = tf.expand_dims(action_prev,0)

		# predict encoding from previous image encoding and state
		image_enc_pred = self.model_forward([self.state, action_prev], training=False)

		# update image encoding and state
		self.image_enc = self.model_image_encoder(tf.expand_dims(image, 0), training=False)
		self.state = self.model_state([self.state, self.image_enc], training=False)

		# visual log output
		if self.output_visual_log:
			self.image_flats.append(tf.reduce_mean(image[:,:,0:3], axis=0).numpy())
			self.image_encs.append((self.image_enc[0]*0.5 + 0.5).numpy())
			self.image_enc_preds.append((image_enc_pred[0]*0.5 + 0.5).numpy())
			self.states.append((self.state[0]*0.5 + 0.5).numpy())

		# return curiosity reward - difference between predicted and real image encoding
		return tf.reduce_mean(tf.abs(self.image_enc[0] - image_enc_pred[0]))

	"""
	Reset state (after an episode)
	"""
	def reset_state(self):
		self.image_enc = tf.zeros((1, self.image_enc_size))
		self.state = tf.zeros((1, self.state_size))

	"""
	Predict action from the state of the model
	"""
	def predict_action(self, model_id, epsilon=0.0):
		state_input = (1.0-epsilon)*self.state +\
			epsilon*tf.random.uniform((1, self.state_size), -1.0, 1.0)
		action = self.models_action[model_id](state_input, training=False)[0]

		return action.numpy()

	def predict_worst_action(self):
		action = tf.expand_dims(tf.random.normal([15], mean=0.0, stddev=0.01), 0)

		for i in range(10):
			with tf.GradientTape() as g:
				g.watch(action)
				reward = self.model_reward([self.state, action], training=False)
			action_grad = g.gradient(reward, action)
			action_grad /= tf.math.reduce_std(action_grad) + 1e-8
			action = action - action_grad*self.action_predict_step_size
		
		action_max = tf.math.reduce_max(tf.abs(action))
		self.action_predict_step_size = ((1.0 / action_max)*0.1 + 0.9)*self.action_predict_step_size

		return convert_action_to_mixed(action[0])


	def train(self, memory):
		n_sequences = memory.images.shape[1]
		# image_encs = tf.Variable(tf.zeros((self.replay_sample_length, n_sequences,
		# 	self.image_enc_size)))

		num_epochs_trained = self.num_epochs_init

		for e in range(self.n_training_epochs):
			images, actions, rewards, state_init = memory.get_sample(self.replay_sample_length)

			# unused
			# state_prev = state_init
			loss_total = 0.0
			loss_regularizer = 0.0
			double_edec_loss = 0.0
			g_model_image_encoder = None
			g_model_image_decoder = None
			n_iters = 0
			for i in range(self.replay_sample_length):
				image_pred, depth_pred, gi_model_image_encoder, gi_model_image_decoder,\
					loss_total_tf, loss_regularizer_tf = self.train_autoencoder(images, tf.convert_to_tensor(i))
				
				loss_total += sum([l.numpy() for l in loss_total_tf]) / len(loss_total_tf)
				loss_regularizer += sum([l.numpy() for l in loss_regularizer_tf]) / len(loss_regularizer_tf)
				
				if g_model_image_encoder is None:
					g_model_image_encoder = gi_model_image_encoder
				else:
					g_model_image_encoder = [a+b for a,b in zip(g_model_image_encoder, gi_model_image_encoder)]

				if g_model_image_decoder is None:
					g_model_image_decoder = gi_model_image_decoder
				else:
					g_model_image_decoder = [a+b for a,b in zip(g_model_image_decoder, gi_model_image_decoder)]

				n_iters += 1
				if not self.quiet:
					image = images[i][:,:,:,1:]
					image_pred_stacked = layers.Concatenate(axis=3)([image_pred, depth_pred])

					# Encode+decode twice (with the same automap) and check the yuv results
					image_target = images[i][:,:,:,1:4]
					automap = images[i][:,:,:,0:1]
					image_target_combined = images[i][:,:,:,1:]
					double_edec_loss += metrics.double_edec_loss(self.model_image_encoder,
					self.model_image_decoder, image_target, image_target_combined, automap)

					print("Epoch {:3d} - Training autoenc. model ({}/{}) l_t: {:8.5f} l_r: {:8.5f} l_edec: {:8.5f}\r".format(
						e, i, self.replay_sample_length,
						loss_total/n_iters,
						loss_regularizer/n_iters,
						double_edec_loss/n_iters), end=" ")
						
					image_loss = ImageLoss(image, image_pred_stacked)
					cv2.imshow("mask", image_loss.loss_mask[e%self.n_replay_episodes].numpy())
					cv2.imshow("mask_gradient", image_loss.loss_mask_gradient[e%self.n_replay_episodes].numpy())
					# print(tf.math.multiply(image_loss.losses, image_loss.weight_matrix).numpy())

					yuv_target = image[e%self.n_replay_episodes,:,:,0:3]
					yuv_pred = image_pred_stacked[e%self.n_replay_episodes,:,:,0:3]
					show_frame_comparison_all(yuv_to_rgb(yuv_target),
						yuv_to_rgb(yuv_pred),
						"rgb")

					show_frame_comparison_all(
						image[e%self.n_replay_episodes,:,:,3:4],
						image_pred_stacked[e%self.n_replay_episodes,:,:,3:4],
						"depth"
					)
					cv2.waitKey(1)

			if not self.quiet:
				# Metrics
				metrics.print_total_metrics(yuv_target, yuv_pred)
				
				print("")

			self.optimizer.learning_rate.assign(learning_rate(num_epochs_trained, 64, 1024, 0.01, 0.001))

			# Update Tensorboard
			with self.train_summary_writer.as_default():
				tf.summary.scalar('loss_total', loss_total / n_iters, step=num_epochs_trained)

			self.optimizer.apply_gradients(zip(g_model_image_encoder,
				self.model_image_encoder.trainable_variables))
			self.optimizer.apply_gradients(zip(g_model_image_decoder,
				self.model_image_decoder.trainable_variables))

			branch = str(self.repo.commit('HEAD'))
			print("branch:", branch)
			branch = branch[0:7]
			num_epochs_trained += 1
			self.save_model(self.model_directory, f"model-{branch}-{num_epochs_trained}")



			del images, actions, rewards, state_init
			gc.collect()

		# Update this number
		# Otherwise the file naming will suck
		self.num_epochs_init = num_epochs_trained
	
	#@tf.function
	def train_image_autoencoder(self, image):
		n_epochs = 8

		n_entries = image.shape[0]

		image = tf.convert_to_tensor(image)
		for e in range(n_epochs):
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


	def save_model(self, folder_name, model_name):
		if not os.path.exists(folder_name):
			os.makedirs(folder_name)
		if not os.path.exists(f"{folder_name}_backup"):
			os.makedirs(f"{folder_name}_backup")

		# backup
		ret = os.system(f"rm {folder_name}_backup/*.h5")
		ret = os.system(f"mv {folder_name}/* {folder_name}_backup/")

		print("Saving model with prefix: {}/{}".format(folder_name, model_name))
		self.model_image_encoder.save_weights("{}/{}_image_encoder.h5".format(folder_name, model_name))
		self.model_image_decoder.save_weights("{}/{}_image_decoder.h5".format(folder_name, model_name))
		# self.model_state.save_weights("{}/{}_state.h5".format(folder_name, model_name))
		# for i in range(self.n_replay_episodes):
		# 	self.models_action[i].save("{}/{}_action_{}.h5".format(folder_name, model_name, i))
		# self.model_reward.save_weights("{}/{}_reward.h5".format(folder_name, model_name))
		# self.model_forward.save_weights("{}/{}_forward.h5".format(folder_name, model_name))
		# self.model_inverse.save_weights("{}/{}_inverse.h5".format(folder_name, model_name))
		# self.model_inverse_backbone.save_weights("{}/{}_inverse_backbone.h5".format(folder_name, model_name))
	
	def load_with_backup(self, model, filename, backup_filename):
		try:
			model.load_weights(filename)
		except:
			try:
				model.load_weights(backup_filename)
			except:
				print(f"Warning: Could not load model from {filename} or {backup_filename}")
				pass
		
	def load_model(self, folder_name, model_name):
		self.branch_id = model_name.split("-")[1]
		self.num_epochs_init = int(model_name.split("-")[2])

		print("Loading model: {}/{}".format(folder_name, model_name))
		self.load_with_backup(self.model_image_encoder,
			"{}/{}_image_encoder.h5".format(folder_name, model_name),
			"{}_backup/{}_image_encoder.h5".format(folder_name, model_name))
		self.load_with_backup(self.model_image_decoder,
			"{}/{}_image_decoder.h5".format(folder_name, model_name),
			"{}_backup/{}_image_decoder.h5".format(folder_name, model_name))
		# self.load_with_backup(self.model_state,
		# 	"{}/{}_state.h5".format(folder_name, model_name),
		# 	"{}_backup/{}_state.h5".format(folder_name, model_name))
		# self.load_with_backup(self.model_reward,
		# 	"{}/{}_reward.h5".format(folder_name, model_name),
		# 	"{}_backup/{}_reward.h5".format(folder_name, model_name))
		# self.load_with_backup(self.model_forward,
		# 	"{}/{}_forward.h5".format(folder_name, model_name),
		# 	"{}_backup/{}_forward.h5".format(folder_name, model_name))
		# self.load_with_backup(self.model_inverse,
		# 	"{}/{}_inverse.h5".format(folder_name, model_name),
		# 	"{}_backup/{}_inverse.h5".format(folder_name, model_name))
		# self.load_with_backup(self.model_inverse_backbone,
		# 	"{}/{}_inverse_backbone.h5".format(folder_name, model_name),
		# 	"{}_backup/{}_inverse_backbone.h5".format(folder_name, model_name))

		# for i in range(self.n_replay_episodes):
		# 	self.load_with_backup(self.models_action[i].model_action,
		# 		"{}/{}_action_{}.h5".format(folder_name, model_name, i),
		# 		"{}_backup/{}_action_{}.h5".format(folder_name, model_name, i))
	
	def create_copy(self):
		model_copy = Model(
			self.episode_length,
			self.n_replay_episodes,
			self.n_training_epochs,
			self.replay_sample_length,
			self.train_log_dir)
		
		model_copy.model_image_encoder.set_weights(self.model_image_encoder.get_weights())
		model_copy.model_image_decoder.set_weights(self.model_image_decoder.get_weights())
		# obsolete, not in the model anymore... right?
		# model_copy.model_state.set_weights(self.model_state.get_weights()) 
		# for i in range(self.n_replay_episodes):
		# 	model_copy.models_action[i].model_action.set_weights(self.models_action[i].model_action.get_weights())
		# model_copy.model_reward.set_weights(self.model_reward.get_weights())
		# model_copy.model_forward.set_weights(self.model_forward.get_weights())
		# model_copy.model_inverse.set_weights(self.model_inverse.get_weights())
		# model_copy.model_inverse_backbone.set_weights(self.model_inverse_backbone.get_weights())

		return model_copy