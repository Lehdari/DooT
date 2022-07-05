import numpy as np
from numpy_ringbuffer import RingBuffer
import random
import tensorflow as tf
import os
from pathlib import Path


class Memory:
    def __init__(self, n_episodes, episode_length, discount_factor=0.995, use_ringbuffer=False):
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.discount_factor = discount_factor
        self.use_ringbuffer = use_ringbuffer
        self.state_size = 512 # model internal state size

        self.clear()


    def clear(self):
        if self.use_ringbuffer:
            self.images = [RingBuffer(capacity=self.episode_length, dtype=object)
                for i in range(self.n_episodes)]
            self.actions = [RingBuffer(capacity=self.episode_length, dtype=object)
                for i in range(self.n_episodes)]
            self.rewards = [RingBuffer(capacity=self.episode_length, dtype=float)
                for i in range(self.n_episodes)]
        else:
            self.images = np.zeros((self.episode_length, self.n_episodes, 240, 320, 5), dtype=np.uint8)
            self.actions = np.zeros((self.episode_length, self.n_episodes, 15), dtype=np.float32)
            self.rewards = np.zeros((self.episode_length, self.n_episodes), dtype=np.float32)

        self.states = np.zeros((self.episode_length, self.n_episodes, self.state_size),
            dtype=np.float32)

        self.episode_lengths = np.zeros((self.n_episodes,), dtype=int)
        self.active_episode = 0


    def store_entry(self, time_step, image, action, reward):
        if self.use_ringbuffer:
            self.images[self.active_episode].append(image)
            self.actions[self.active_episode].append(action)
            self.rewards[self.active_episode].append(reward)

            self.episode_lengths[self.active_episode] =\
                min(self.episode_length, time_step+1)
        else:
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
            if self.use_ringbuffer:
                # convert ringbuffers to tensors
                self.images = np.transpose(np.array([np.stack(i) for i in self.images]), axes=(1,0,2,3,4))
                self.actions = np.transpose(np.array([np.stack(i) for i in self.actions]), axes=(1,0,2))
                self.rewards = np.transpose(np.array([np.stack(i) for i in self.rewards]), axes=(1,0))
            self.discount_rewards()

        return memory_full
    

    def compute_states(self, model_state, model_image_encoder, end=None):
        if end is None:
            end = np.amin(self.episode_lengths)
        state = tf.zeros((self.n_episodes, self.state_size))
        for i in range(end):
            state = model_state([state, model_image_encoder(self.images[i], training=False)],
                training=False).numpy()
            self.states[i] = state
            print("Computing states... ({}/{})".format(i, end), end="\r")


    @staticmethod
    def rand_int_range_or_zero(x):
        if x == 0:
            return 0
        else:
            random.randint(0, x)


    def get_sample(self, length, model_state=None, model_image_encoder=None):
        # min_episode_length = np.amin(self.episode_lengths)

        print("== get_sample() ===")

        print(f"Episode lengths: {self.episode_lengths}")

        begin = np.array([Memory.rand_int_range_or_zero(l-length)
            for l in self.episode_lengths])
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


    def save(self, filename):
        """
        Save memory in compressed .npz format
        """
        directory = Path(filename).parents[0]
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(Path(filename), "wb") as f:
            np.savez_compressed(f,
                images=self.images,
                actions=self.actions,
                rewards=self.rewards,
                episode_lengths=self.episode_lengths
            )

    def load(self, filename):
        """
        Load memory from compressed .npz file
        """
        if not os.path.exists(Path(filename)):
            print(f"Memory: Unable to load from {filename}, file does not exist.")
        
        loaded = np.load(Path(filename))

        assert "images" in loaded
        assert "actions" in loaded
        assert "rewards" in loaded
        assert "episode_lengths" in loaded

        self.images = loaded["images"]
        self.actions = loaded["actions"]
        self.rewards = loaded["rewards"]
        self.episode_lengths = loaded["episode_lengths"]
        self.episode_length = self.images.shape[0]
        self.n_episodes = self.images.shape[1]
