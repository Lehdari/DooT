from reward import Reward
from model import Model
import numpy as np
import tensorflow as tf
import random
import math
import time
from scipy.signal import argrelextrema
from utils import *

import matplotlib.pyplot as plt


class MemoryEntry:
	"""
	Create memory entry
	state_prev:		Input state (previous state)
	image_in:		Input image
	action_prev:	Previous action
	action:			Action
	reward:			Reward given after game state has been advanced using action
	"""
	def __init__(self, image, action, reward):
		self.image = image
		self.action = convert_action_to_continuous(action) # convert action into continuous domain
		self.reward = reward
		self.reward_disc = reward # discounted reward


class MemorySequence:
	def __init__(self, discount_factor=0.97):
		self.sequence = []
		self.discount_factor = discount_factor
		self.reward_cum = 0.0 # cumulative reward, assigned in the end of episode

	def add_entry(self, image, action, reward):
		self.sequence.append(MemoryEntry(image, action, reward))

	def discount(self):
		# iterate the sequence backwards to pass discounted rewards to previous entries
		for i in range(len(self.sequence)-2, -1, -1):
			self.sequence[i].reward_disc =\
				self.discount_factor*self.sequence[i+1].reward_disc +\
				(1.0-self.discount_factor)*self.sequence[i].reward
			# future rewards can only affect positively
			if self.sequence[i].reward_disc < self.sequence[i].reward:
				self.sequence[i].reward_disc = self.sequence[i].reward
		
		# rewards = []
		# rewards_disc = []
		# for e in self.sequence:
		# 	rewards.append(e.reward)
		# 	rewards_disc.append(e.reward_disc)
		
		# x = np.linspace(0, len(self.sequence), len(self.sequence), endpoint=False)
		# plt.plot(x, rewards)
		# plt.plot(x, rewards_disc)
		# plt.show()

	def get_best_entries(self, n_entries):
		# sort according to discounted reward
		seq_sorted = sorted(self.sequence, key=lambda entry: entry.reward_disc)
		return seq_sorted[-n_entries:]
	
	def get_entries_threshold(self, threshold):
		return filter(lambda e: e.reward_disc > threshold, self.sequence)
	
	# find best clutch moment (subsequence with highest reward increase rate)
	def get_best_clutch(self, smooth_size=128):
		reward = [e.reward for e in self.sequence]
		
		# smoothing
		sigma = smooth_size / 3.3
		smooth_kernel = np.linspace(-3.3, 3.3, smooth_size+1)
		smooth_kernel = (1.0/(sigma*np.sqrt(2.0*math.pi)))*\
			np.exp(-0.5*np.power(smooth_kernel, 2.0))
		reward_smooth = np.convolve(reward, smooth_kernel, mode="same")

		# list all extremum points
		extrema = np.sort(np.concatenate([np.array([[0]]),
			argrelextrema(reward_smooth, np.greater),
			argrelextrema(reward_smooth, np.less),
			np.array([[len(reward_smooth)-1]])], axis=1).flatten())

		# find the two consecutive extrema with highest reward increase rate
		best_extrema = [-1, -1]
		slope_max = 0.0
		for i in range(extrema.shape[0]-1):
			slope = (reward_smooth[extrema[i+1]] - reward_smooth[extrema[i]]) /\
				(extrema[i+1] - extrema[i])
			if slope > slope_max:
				best_extrema = [extrema[i], extrema[i+1]]
				slope_max = slope

		if best_extrema[0] < 0:
			return [] # only downhill from the beginning :(
		else:
			return self.sequence[best_extrema[0]:best_extrema[1]]
	
	def get_average_discounted_reward(self):
		# sort according to discounted reward
		s = 0.0
		for e in self.sequence:
			s += e.reward_disc
		return s / len(self.sequence)


class TrainerInterface:
	def __init__(self, model, reward, episode_length):
		self.model = model
		self.reward = reward

		self.episode_id = 0
		self.episode_reset()

		self.replay_reset()
		self.set_episode_length(episode_length)

	def set_episode_length(self, episode_length):
		self.episode_length = episode_length
		self.replay_images = tf.Variable(tf.zeros((episode_length, 240, 320, 3)))
		self.replay_actions = tf.Variable(tf.zeros((episode_length, 15)))
		self.replay_rewards_step = tf.Variable(tf.zeros((episode_length,)))
		self.replay_rewards_avg = tf.Variable(tf.zeros((episode_length,)))
	
	"""
	Reset after an episode
	"""
	def episode_reset(self):
		self.reward.reset()
		self.model.reset_state()
		self.action_prev = get_null_action()

		self.memory = MemorySequence()
		self.reward_cum = 0.0 # cumulative reward
		self.reward_n = 0
	
	"""
	Reset after an experience replay
	"""
	def replay_reset(self):
		# lists for training entries
		#self.replay_images = []
		#self.replay_actions = []
		#self.replay_rewards = []
		self.n_entries = 0
	
	def save_replay_entry(self, entry):
		self.replay_images.append(entry.image)
		self.replay_actions.append(entry.action)
		self.replay_rewards.append(entry.reward_disc)


	"""
	Pick an action to perform next

	Override this method in child classes to implement more complex decision process
	"""
	def pick_action(self, game):
		# pick an action to perform
		return get_random_action(weapon_switch_prob=0.03)

	def pick_top_replay_entries(self):
		return self.memory.get_best_entries(int(len(self.memory.sequence)/2))
	
	def mix_reward(self, reward_model, reward_game, reward_system):
		return reward_model + reward_game + reward_system

	def step(self, game, episode_id, frame_id):
		state_game = game.get_state()
		self.episode_id = episode_id

		screen_buf = state_game.screen_buffer

		# pick an action to perform
		action = self.pick_action(game)

		# action_print = np.where(action, 1, 0)
		# print("{} {:8.3f}".format(action_print[0:14], action[14]), end="\r")
		# time.sleep(2.0)

		# advance the model state using the screen buffer
		reward_model = self.model.advance(screen_buf, action)

		# Only pick up the death penalty from the builtin reward system
		reward_game = game.make_action(action)

		# Fetch rest of the rewards from our own reward system
		reward_system = self.reward.get_reward(game, action)

		reward = self.mix_reward(reward_model, reward_game, reward_system)

		# update cumulative reward
		self.reward_cum += reward;
		self.reward_n += 1

		# TODO temp
		action_print = np.where(action, 1, 0)
		print("{} {:8.3f} | {:8.3f}".format(action_print[0:14], action[14], reward), end="\r")
		# TODO end of temp

		# Save the step into the memory
		#self.memory.add_entry(screen_buf, action, reward)

		self.replay_images[frame_id].assign(screen_buf)
		self.replay_actions[frame_id].assign(convert_action_to_continuous(action))
		self.replay_rewards_step[frame_id].assign(reward)
		self.n_entries += 1

		done = game.is_episode_finished()
		if done:
			# save the cumulative reward to the sequence
			self.memory.reward_cum = self.reward_cum

			# calculate discounted rewards for the memory sequence
			self.memory.discount()
			
			# entries with highest discounted value
			#top_entries = self.pick_top_replay_entries()

			print("\nEpisode {} finished\naverage reward: {:10.3f}"
				.format(episode_id, self.reward_cum / self.episode_length))

			# add top entries to the training buffers
			#for e in top_entries:
			#	self.save_replay_entry(e)

			# Sufficient number of entries gathered, time to train
			if self.n_entries == self.episode_length:
				# average reward
				self.replay_rewards_avg = tf.ones((self.episode_length,)) *\
					(self.reward_cum / self.episode_length)
				# train
				self.model.train(self.replay_images, self.replay_actions, self.replay_rewards_step,
					self.replay_rewards_avg)
			
			# reset stuff for the new episode
			self.replay_reset()
			self.episode_reset()
			self.reward.reset_exploration()
