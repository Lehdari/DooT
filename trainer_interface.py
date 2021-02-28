from reward import Reward
from model import Model
from memory import Memory
import numpy as np
import tensorflow as tf
import random
import math
import time
from scipy.signal import argrelextrema
from utils import *

import matplotlib.pyplot as plt


class TrainerInterface:
	def __init__(self, model, reward, n_episodes, episode_length, minimum_episode_length):
		self.model = model
		self.reward = reward
		self.memory = Memory(n_episodes, episode_length)

		self.replay_reset()

		self.episode_id = 0
		self.n_replay_episodes = n_episodes
		self.episode_length = episode_length
		self.minimum_episode_length = minimum_episode_length
		self.episode_reset()

	"""
	Reset after an episode
	"""
	def episode_reset(self):
		self.reward.reset()
		self.model.reset_state()
		self.action_prev = get_null_action()

		self.reward_cum = 0.0 # cumulative reward
		self.n_entries = 0
	
	"""
	Reset after an experience replay
	"""
	def replay_reset(self):
		self.n_entries = 0
		self.memory.clear()

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
		# advance the model state using the screen buffer
		reward_model = self.model.advance(screen_buf, self.action_prev).numpy()
		#reward_model = 0.0
		
		# pick an action to perform
		action = self.pick_action(game)
		self.action_prev = action # store the action for next step

		# Only pick up the death penalty from the builtin reward system
		reward_game = game.make_action(action)

		# Fetch rest of the rewards from our own reward system
		reward_system = self.reward.get_reward(game, action)

		reward = self.mix_reward(reward_model, reward_game, reward_system)

		# update cumulative reward
		self.reward_cum += reward

		# TODO temp
		action_print = np.where(action, 1, 0)
		print("{} {:8.3f} | r: {:3.8f} e: {:2.8f}".format(
			action_print[0:14], action[14], reward, self.epsilon), end="\r")
		# TODO end of temp

		# Save the step into the memory
		self.memory.store_entry(self.n_entries,
			screen_buf, convert_action_to_continuous(action), reward)

		self.n_entries += 1

		done = game.is_episode_finished()
		if done:
			print("\nEpisode {} finished\naverage reward: {:10.3f}"
				.format(episode_id, self.reward_cum / self.n_entries))
			
			# overwrite last if minimum episode length was not reached
			if self.n_entries < self.minimum_episode_length:
				print("Episode underlength ({}), discarding...".format(self.n_entries))
				self.episode_reset()
				return

			# Sufficient number of entries gathered, time to train
			if self.memory.finish_episode():
				self.model.train(self.memory)

				self.replay_reset()
			
			# reset stuff for the new episode
			self.episode_reset()
			self.reward.reset_exploration()
