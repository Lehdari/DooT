import numpy as np
import random
import tensorflow as tf


class Memory:
    def __init__(self, n_episodes, episode_length, discount_factor=0.995):
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.discount_factor = discount_factor
        self.state_size = 256 # model internal state size

        self.clear()


    def clear(self):
        self.images = np.zeros((self.episode_length, self.n_episodes, 240, 320, 4), dtype=np.uint8)
        self.actions = np.zeros((self.episode_length, self.n_episodes, 15), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_episodes), dtype=np.float32)

        self.states = np.zeros((self.episode_length, self.n_episodes, self.state_size),
            dtype=np.float32)

        self.episode_lengths = np.zeros((self.n_episodes,), dtype=int)
        self.active_episode = 0


    def store_entry(self, time_step, image, action, reward):
        self.images[time_step, self.active_episode] = image
        self.actions[time_step, self.active_episode] = action
        self.rewards[time_step, self.active_episode] = reward

        # keep track of last time step for each episode due to premature termination
        # (death or level finish)
        self.episode_lengths[self.active_episode] =\
            max(self.episode_lengths[self.active_episode], time_step+1)
    

    def discount_rewards(self):
        # normalization parameter for rewards
        discount_scale = -np.log(self.discount_factor)
        for i in range(self.n_episodes):
            reward_cum = 0.0
            for j in range(self.episode_length-1, -1, -1):
                reward_cum = reward_cum*self.discount_factor + self.rewards[j, i]*discount_scale
                self.rewards[j, i] = reward_cum
    

    def finish_episode(self):
        self.active_episode += 1

        memory_full = self.active_episode == self.n_episodes

        if memory_full:
            self.discount_rewards()

        return memory_full
    

    def compute_states(self, model_state, model_image_encoder, end=None):
        if end is None:
            end = np.amin(self.episode_lengths)
        state = tf.zeros((self.n_episodes, self.state_size))
        action = tf.zeros(shape=(8,15), dtype=tf.float32)
        for i in range(end):
            state = model_state([state, model_image_encoder(self.images[i], training=False), action],
                training=False).numpy()
            action = self.actions[i]
            self.states[i] = state
            print("Computing states... ({}/{})".format(i, end), end="\r")


    def get_sample(self, length, model_state=None, model_image_encoder=None):
        # min_episode_length = np.amin(self.episode_lengths)
        begin = np.array([random.randint(0, l-length-1) for l in self.episode_lengths])
        #end = begin + length

        if model_state is not None and model_image_encoder is not None:
            self.compute_states(model_state, model_image_encoder, np.amax(begin))
        
        state = np.zeros((self.n_episodes, self.state_size), dtype=np.float32)

        for i in range(self.n_episodes):
            state[i] = self.states[begin[i], i]
        
        i = np.arange(self.n_episodes)
        j = np.repeat(np.expand_dims(np.arange(length), 1), self.n_episodes, axis=1)
        
        images_slice = self.images[begin[np.newaxis,:]+j, i[np.newaxis,:]]
        actions_slice = self.actions[begin[np.newaxis,:]+j, i[np.newaxis,:]]
        rewards_slice = self.rewards[begin[np.newaxis,:]+j, i[np.newaxis,:]]

        return\
            (tf.convert_to_tensor(images_slice, dtype=tf.float32) * 0.0039215686274509803,
            tf.convert_to_tensor(actions_slice),
            tf.convert_to_tensor(rewards_slice),
            tf.convert_to_tensor(state))