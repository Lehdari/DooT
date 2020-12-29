from reward import Reward
from model import Model
import numpy as np
import tensorflow as tf
import random
from utils import *


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
		self.disc_reward = reward # discounted reward


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
		# print("Entry {:3d}: r: {:10.5f} dr: {:10.5f}".format(len(self.sequence)-1,
		# 	self.sequence[-1].reward, self.sequence[-1].disc_reward))
		for i in range(len(self.sequence)-2, -1, -1):
			self.sequence[i].disc_reward =\
				self.discount_factor*self.sequence[i+1].disc_reward +\
				self.sequence[i].reward

			# print("Entry {:3d}: r: {:10.5f} dr: {:10.5f}".format(i, self.sequence[i].reward,
			# 	self.sequence[i].disc_reward))

	def get_best_entries(self, n_entries):
		# sort according to discounted reward
		seq_sorted = sorted(self.sequence, key=lambda entry: entry.disc_reward)
		return seq_sorted[-n_entries:]


class Trainer:
	def __init__(self, model, reward):
		self.model = model
		self.reward = reward

		self.memory = [] # list of sequences
		self.replay_episode_interval = 8 # experience replay interval in episodes
		self.replay_n_sequences = 4 # use this many best sequences in experience replay
		self.replay_n_entries = 64 # number of entries per seq. to use in exp. replay
		self.gamma = 0.85
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.99999
		self.learning_rate = 0.005

		self.action_prev = np.zeros((15,))
		self.episode_id_prev = -1
		self.reward_cum = 0.0 # cumulative reward

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
		if episode_id != self.episode_id_prev:
			self.memory.append(MemorySequence())
			self.episode_id_prev = episode_id

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
		self.epsilon *= self.epsilon_decay
		self.epsilon = max(self.epsilon_min, self.epsilon)
		#print("epsilon: {}".format(self.epsilon)) # TODO remove
		if np.random.random() < self.epsilon:
			action = self.model.get_random_action() # make random action
		else:
			action = self.model.predict_action() # make action predicted from model state

		# Intentionally ignore the reward the game gives
		game.make_action(action)

		# Instead, use our own reward system
		reward = self.reward.get_reward(game)
		self.reward_cum += reward;

		# Save the step into active(last in the list) memory sequence
		self.memory[-1].add_entry(screen_buf, state_prev, self.action_prev, action, reward)

		# save the action taken for next step
		self.action_prev = action.copy()

		# print("reward: {}".format(reward)) # TODO remove

		done = game.is_episode_finished()
		if done:
			print("Episode {} finished, cumulative reward: {:10.5f}, epsilon: {:10.5f}"
				.format(episode_id, self.reward_cum, self.epsilon))
			# save the cumulative reward to the sequence
			self.memory[-1].reward_cum = self.reward_cum
			self.reward_cum = 0.0

			# reset stuff for the new episode
			self.action_prev = np.zeros((15,))
			self.reward.reset()
			self.model.reset_state()

			if (episode_id+1) % self.replay_episode_interval == 0:
				print("================================================================================")
				print("Experience replay interval reached, training...")

				# gather best entries from the memory
				frames_in = []
				states_in = []
				actions_in = []
				actions_out = []

				# use self.replay_n_sequences best sequences
				self.memory.sort(key=lambda sequence: sequence.reward_cum)
				for sequence in self.memory[-self.replay_n_sequences:]:
					print(sequence.reward_cum)
					sequence.discount()
					best = sequence.get_best_entries(self.replay_n_entries)
					for e in best:
						frames_in.append(e.frame_in)
						states_in.append(e.state_in)
						actions_in.append(e.action_in)
						actions_out.append(e.action_out)

				# train
				self.model.train(frames_in, states_in, actions_in, actions_out)

				# clear memory
				self.memory = []
