from reward import Reward
from model import Model
import numpy as np
import tensorflow as tf
from collections import deque
import random


class Trainer:
	def __init__(self, model, reward):
		self.model = model
		self.reward = reward

		self.memory = deque(maxlen=2000)
		self.gamma = 0.85
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.005

		self.action_prev = np.zeros((15,))

	"""
	Perform one step;
	- create action
	- get reward
	- update game state (happens in make_action under the hood)

	return: reward (1D float)
	"""
	def step(self, game):
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
		if np.random.random() < self.epsilon:
			action = self.model.get_random_action() # make random action
		else:
			action = self.model.predict_action() # make action predicted from model state

		# Intentionally ignore the reward the game gives
		game.make_action(action)

		# Instead, use our own reward system
		reward = self.reward.get_reward(game)

		# save the action taken for next step
		self.action_prev = action.copy()

		print("reward: {}".format(reward))

		# done = game.is_episode_finished()
		# bufu = None
		# if not done:
		# 	bufu = game.get_state().screen_buffer
		# self.remember(screen_buf, action, reward, bufu, done)

		#self.replay()

		return reward


	"""
	Add entry to memory
	"""	
	def remember(self, state, action, reward, new_state, done):
		self.memory.append([state, action, reward, new_state, done])

	def replay(self):
		batch_size = 32
		if len(self.memory) < batch_size: 
			return

		samples = random.sample(self.memory, batch_size)
		for sample in samples:
			state, action, reward, new_state, done = sample
			target = self.model.predict(state)
			if done:
				target[0][action] = reward
			else:
				# Maximum q-value for s'
				Q_future = max(self.model.predict(new_state)[0])
				target[0][action] = reward + Q_future * self.gamma
			self.model.fit(state, target, epochs=1, verbose=0)