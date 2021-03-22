from reward import Reward
from model import Model
from memory import Memory
import numpy as np
import tensorflow as tf
import random
import math
import time
from random import choice
from utils import *
from generate_maps import *
from init_game import init_game
import cv2


class TrainerInterface:
	def __init__(self, reward, n_episodes, episode_length, minimum_episode_length,
		window_visible):
		self.reward = reward

		self.episode_id = 0
		self.n_replay_episodes = n_episodes
		self.episode_length = episode_length
		self.minimum_episode_length = minimum_episode_length
		self.window_visible = window_visible
		self.n_discards = 0

	"""
	Reset after an episode
	"""
	def episode_reset(self, model):
		self.reward.reset()
		model.reset_state()
		self.action_prev = get_null_action()

		self.reward_cum = 0.0 # cumulative reward
		self.n_entries = 0

	"""
	Pick an action to perform next

	Override this method in child classes to implement more complex decision process
	"""
	def pick_action(self, game, model):
		# pick an action to perform
		return get_random_action(weapon_switch_prob=0.03)

	def pick_top_replay_entries(self):
		return self.memory.get_best_entries(int(len(self.memory.sequence)/2))
	
	def mix_reward(self, reward_model, reward_game, reward_system):
		return reward_model + reward_game + reward_system
	
	def generate_new_maps(self, game):
		game.close()
		generate_maps(seed=random.randint(0, 999999999999))
		game.set_doom_scenario_path("wads/temp/oblige.wad")
		game.init()
	
	def run(self, model):
		game = init_game(self.episode_length, self.window_visible)

		map_names = ["map01", "map02", "map03", "map04", "map05",
			"map06", "map07", "map08", "map09", "map10",
			"map11", "map12", "map13", "map14", "map15",
			"map16", "map17", "map18", "map19", "map20"]
		
		self.memory = Memory(self.n_replay_episodes, self.episode_length, discount_factor=0.98)
		self.generate_new_maps(game)

		while True:
			if self.n_discards >= 10: # generate new maps if some of the current ones proves too difficult
				self.generate_new_maps(game)
			
			game.set_doom_map(map_names[self.episode_id%self.n_replay_episodes])
			game.new_episode()

			# setup automap scale
			game.send_game_command('am_scale 0.5')

			self.episode_reset(model)
			self.reward.player_start_pos = get_player_pos(game)

			frame_id = 0
			while not game.is_episode_finished():
				if self.step(game, model, frame_id):
					# cv2.destroyWindow("ViZDoom Automap")
					# cv2.waitKey(1)
					game.close()

					return self.memory

				frame_id += 1

	def step(self, game, model, frame_id):
		state_game = game.get_state()

		automap = state_game.automap_buffer[:,:,0:1] # use the red channel, should be enough
		screen_buf = np.concatenate([state_game.screen_buffer, automap], axis=-1)
		# if self.window_visible:
		# 	cv2.imshow("ViZDoom Automap", automap)
		# 	cv2.waitKey(1)
		
		#with tf.device("/cpu:0"):
		# advance the model state using the screen buffer
		reward_model = model.advance(screen_buf, self.action_prev).numpy()
		# reward_model = 0.0
		
		# pick an action to perform
		action = self.pick_action(game, model)

		self.action_prev = action # store the action for next step

		# Only pick up the death penalty from the builtin reward system
		reward_game = game.make_action(convert_action_to_mixed(action))

		# Fetch rest of the rewards from our own reward system
		reward_system = self.reward.get_reward(game, action)

		reward = self.mix_reward(reward_model, reward_game, reward_system)

		# update cumulative reward
		self.reward_cum += reward

		# TODO temp
		# action_print = np.where(action>0.0, 1, 0)
		# print("{} {:8.3f} | r: {:3.8f} e: {:2.8f}".format(
		# 	action_print[0:14], action[14]*10.0, reward, self.epsilon), end="\r")
		# TODO end of temp

		# Save the step into the memory
		self.memory.store_entry(self.n_entries, screen_buf, action, reward)
		self.n_entries += 1

		done = game.is_episode_finished()
		if done:
			print("\nEpisode {} finished, average reward: {:10.3f}"
				.format(self.episode_id, self.reward_cum / self.n_entries))
			
			# overwrite last if minimum episode length was not reached
			if self.n_entries < self.minimum_episode_length:
				print("Episode underlength ({}), discarding...".format(self.n_entries))
				self.n_discards += 1
				return False
			
			self.episode_id += 1 # don't increase episode id after discarding
			self.n_discards = 0

			# Sufficient number of entries gathered, time to train
			if self.memory.finish_episode():
				return True
		
		return False
