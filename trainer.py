from reward import Reward
from model import Model
import numpy as np
import tensorflow as tf
import random
import math
from scipy.signal import argrelextrema
from utils import *

import matplotlib.pyplot as plt


class MemoryEntry:
	"""
	Create memory entry
	frame_in:	Input frame
	state_in:	Input state (previous state)
	action_in:	Input action (action taken previously)
	action_out:	Output action (action taken to advance game state)
	reward:		Reward given after game state has been advanced using action_out
	"""
	def __init__(self, frame_in, state_in, action_in, action_out, reward):
		self.frame_in = frame_in
		self.state_in = state_in
		# convert actions into continuous domain
		self.action_in = convert_action_to_continuous(action_in)
		self.action_out = convert_action_to_continuous(action_out)
		self.reward = reward
		self.reward_disc = reward # discounted reward


class MemorySequence:
	def __init__(self, discount_factor=0.95):
		self.sequence = []
		self.discount_factor = discount_factor
		self.reward_cum = 0.0 # cumulative reward, assigned in the end of episode

	def add_entry(self, frame_in, state_in, action_in, action_out, reward):
		self.sequence.append(MemoryEntry(
			frame_in, state_in, action_in, action_out, reward))

	def discount(self):
		# iterate the sequence backwards to pass discounted rewards to previous entries
		for i in range(len(self.sequence)-2, -1, -1):
			self.sequence[i].reward_disc =\
				self.discount_factor*self.sequence[i+1].reward_disc +\
				self.sequence[i].reward
			# future rewards can only affect positively
			if self.sequence[i].reward_disc < self.sequence[i].reward:
				self.sequence[i].reward_disc = self.sequence[i].reward

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


class Trainer:
	def __init__(self, model, reward):
		self.model = model
		self.reward = reward

		#self.replay_episode_interval = 8 # experience replay interval in episodes
		#self.replay_n_entries_min = 16 # number of entries used for training from worst sequence
		#self.replay_n_entries_delta = 16 # number of entries to increase for better sequences
		self.replay_n_entries = 2048
		self.threshold = 0.0
		self.replay_reset()
		self.threshold_prev = -1000001.0
		
		self.epsilon = 1.0 # probability for random action
		self.epsilon_min = 0.02
		self.epsilon_decay = 0.98

		self.episode_id_prev = -1
		self.episode_reset()
	
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

		self.reward_lowpass = None
		self.epsilon_reactive = self.epsilon
	
	def replay_reset(self):
		self.frames_in = []
		self.states_in = []
		self.actions_in = []
		self.actions_out = []
		self.n_entries = 0
		# new reward average - used as limit threshold for new replay entries
		self.threshold_prev = self.threshold / self.replay_n_entries
		self.threshold = 0.0

	"""
	Perform one step;
	- create action
	- get reward
	- update game state (happens in make_action under the hood)

	game:		ViZDoom game object
	episode_id:	Id of the currently running episode
	return: 	reward (1D float)
	"""
	def step(self, game, episode_id):
		state_game = game.get_state()

		#TODO: stack frames
		screen_buf = state_game.screen_buffer

		# save the previous model state
		state_prev = self.model.state.copy()

		# advance the model state using new screen buffer and the previously taken action
		self.model.advance(screen_buf, self.action_prev)

		# Epsilon-greedy algorithm
		# With probability epsilon choose a random action ("explore")
		# With probability 1-epsilon choose best known action ("exploit")
		if np.random.random() < self.epsilon_reactive:
			# with 90% change just mutate the previous action since usually in Doom there's
			# strong coherency between consecutive actions
			if np.random.random() > 0.1:
				action = mutate_action(self.action_prev, 2,
					weapon_switch_prob=(0.23-0.2*self.epsilon_reactive))

				# apply some damping to turning delta to reduce that 360 noscope business
				action[14] *= (0.98-0.03*self.epsilon_reactive)
			else:
				action = get_random_action(weapon_switch_prob=(0.45-0.4*self.epsilon_reactive))
		else:
			action = self.model.predict_action() # make action predicted from model state

		# Only pick up the death penalty from the builtin reward system
		reward = game.make_action(action)

		# Instead, use our own reward system
		reward += self.reward.get_reward(game)
		# update cumulative reward and reward delta
		self.reward_cum += reward;
		
		self.reward_n += 1

		if self.reward_lowpass is None:
			self.reward_lowpass = reward
		self.reward_lowpass = 0.97*self.reward_lowpass + 0.03*reward

		if self.reward_lowpass > self.reward_cum / self.reward_n:
			self.epsilon_reactive *= 0.997
		else:
			self.epsilon_reactive += (1.0-self.epsilon_reactive)*0.005

		# TODO temp
		action_print = np.where(action, 1, 0)
		print("{} {:8.3f} | {:8.3f} {:8.3f} | {:8.7f}".format(
			action_print[0:14], action[14], self.reward_lowpass, self.reward_cum/self.reward_n,
			self.epsilon_reactive), end="\r")
		# TODO end of temp

		# Save the step into active(last in the list) memory sequence
		self.memory.add_entry(screen_buf, state_prev, self.action_prev, action, reward)

		# save the action taken for next step
		self.action_prev = action.copy()

		done = game.is_episode_finished()
		if done:
			# save the cumulative reward to the sequence
			self.memory.reward_cum = self.reward_cum
			self.memory.discount()
			
			if self.threshold_prev < -1000000.0:
				# baseline threshold reward
				self.threshold_prev = self.memory.get_average_discounted_reward()
				print("Initial threshold set: {}".format(self.threshold_prev))
			else:
				# entries with highest discounted value
				top_entries = list(self.memory.get_entries_threshold(self.threshold_prev))

				# Reduce threshold and increase random action epsilon if no entries exceed the
				# current threshold. This might be due to drifting to local maximum
				if len(top_entries) == 0:
					self.threshold_prev *= 0.99
					self.epsilon += (1.0-self.epsilon)*0.01

				# pick clutch entries also
				top_entries += self.memory.get_best_clutch()

				self.n_entries += len(top_entries)
				print("Episode {} finished, cumulative reward: {:10.3f}\nepsilon: {:8.5f} entry threshold: {:8.5f} training entries: {}/{}"
					.format(episode_id, self.reward_cum, self.epsilon,\
					self.threshold_prev, self.n_entries, self.replay_n_entries))

				# add top entries to the training buffers
				for e in top_entries:
					self.frames_in.append(e.frame_in)
					self.states_in.append(e.state_in)
					self.actions_in.append(e.action_in)
					self.actions_out.append(e.action_out)
					self.threshold += e.reward_disc

				# Sufficient number of entries gathered, time to train
				if self.n_entries >= self.replay_n_entries:
					# train
					self.model.train(self.frames_in, self.states_in, self.actions_in,
						self.actions_out)
					self.replay_reset()
					self.epsilon *= self.epsilon_decay
					self.epsilon = max(self.epsilon_min, self.epsilon)
			
			# reset stuff for the new episode
			self.episode_reset()
			self.reward.reset_exploration()