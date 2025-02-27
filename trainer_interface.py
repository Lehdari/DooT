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
	def __init__(self, reward: Reward, n_episodes: int, episode_length: int, minimum_episode_length: int,
		window_visible: bool, replay_sample_length: int):
		self.reward = reward

		self.episode_id = 0
		self.n_replay_episodes = n_episodes
		self.episode_length = episode_length
		self.minimum_episode_length = minimum_episode_length
		self.window_visible = window_visible
		self.n_discards = 0
		self.replay_sample_length = replay_sample_length
		self.generated_wad_path = "wads/temp/oblige.wad"
		self.smoketest_wad_path = "wads/smoketest/oblige.wad"
		self.map_names = ["map01", "map02", "map03", "map04", "map05",
			"map06", "map07", "map08", "map09", "map10",
			"map11", "map12", "map13", "map14", "map15",
			"map16", "map17", "map18", "map19", "map20"]
		
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
		game.set_doom_scenario_path(self.generated_wad_path)
		game.init()
	
	def run(self, model, is_smoketest=False, smoketest_length=32) -> Memory:
		game = init_game(self.episode_length, self.window_visible)

		if is_smoketest:
			self.memory = Memory(self.n_replay_episodes, smoketest_length, use_ringbuffer=True)
			game.close()
			game.set_doom_scenario_path(self.smoketest_wad_path)
			game.init()
		else:
			self.memory = Memory(self.n_replay_episodes, self.episode_length, discount_factor=0.98)
			self.generate_new_maps(game)

		while True:
			# generate new maps if some of the current ones proves too difficult
			if self.n_discards >= 10 and not is_smoketest:
				self.generate_new_maps(game)
			
			game.set_doom_map(self.map_names[self.episode_id % self.n_replay_episodes])
			game.new_episode()

			# setup automap scale
			game.send_game_command('am_scale 0.5')
			
			# give all weapons and armor
			game.send_game_command("give health")
			game.send_game_command("give weapons")
			game.send_game_command("give backpack")
			game.send_game_command("give ammo")
			game.send_game_command("give armor")

			# spell of undying
			game.send_game_command("buddha")

			self.episode_reset(model)
			self.reward.player_start_pos = get_player_pos(game)

			frame_id = 0
			while not game.is_episode_finished():
				if self.step(game, model, frame_id): # step returns true when memory is full
					game.close()
					return self.memory

				frame_id += 1

	def step(self, game, model, frame_id) -> bool:
		state_game = game.get_state()

		automap = state_game.automap_buffer[:,:,0:1] # use the red channel, should be enough
		depth = np.expand_dims(state_game.depth_buffer, axis=2)
		screen_buf = np.concatenate([automap, state_game.screen_buffer, depth], axis=-1)
		
		device = "/cpu:0"
		if self.episode_id < self.n_replay_episodes:
			device="/gpu:0"
		with tf.device(device):
			# advance the model state using the screen buffer
			# reward_model = model.advance(screen_buf, self.action_prev).numpy()
			reward_model = 0.0
			
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

		# Save the step into the memory
		self.memory.store_entry(self.n_entries, screen_buf, action, reward)
		self.n_entries += 1

		done = game.is_episode_finished()
		if done:
			print(f"\nEpisode {self.episode_id} finished, average reward: {self.reward_cum / self.n_entries:10.3f} ",
				  f"Episode length: {self.n_entries}")

			# overwrite last if minimum episode length was not reached
			# # TEMP commented away
			# if self.n_entries < self.minimum_episode_length:
			# 	print("Episode underlength ({}), discarding...".format(self.n_entries))
			# 	self.n_discards += 1
			# 	return False

			# TODO: if exit is found really soon we don't want to discard
			# that episode. We need to update model train function
			# to allow having shorter episodes OR append these
			# frames to another entry.
			if self.n_entries < self.replay_sample_length:
				print("Episode was too short")
				self.n_discards +=1
				return False

			self.episode_id += 1 # don't increase episode id after discarding
			self.n_discards = 0

			model.save_episode_state_images(self.episode_id)

			# Sufficient number of entries gathered, time to train
			if self.memory.finish_episode():
				return True
		
		return False
