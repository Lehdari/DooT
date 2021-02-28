from trainer_interface import *
from utils import *
import random
import math


class TrainerSimple(TrainerInterface):
    def episode_reset(self):
        TrainerInterface.episode_reset(self)

        self.epsilon = math.exp(-(self.episode_id)/32.0)
        if self.epsilon < 0.05:
            self.epsilon = 0.05
        
        self.epsilon = 0.0

    def pick_action(self, game):
        r = random.random()
        if r < self.epsilon:
            if random.random() < 0.1:
                action = get_random_action(turn_delta_sigma=5.0,
                    weapon_switch_prob=0.3-0.26*self.epsilon)
                action[14] = 0.5*self.action_prev[14] + 0.5*action[14]
            else:
                action = mutate_action(self.action_prev, 2, turn_delta_sigma=3.0, turn_damping=0.95,
                    weapon_switch_prob=0.3-0.26*self.epsilon)
        else:
            action = self.model.predict_action(self.memory.active_episode)

            if r < self.epsilon*2.0:
                action = mutate_action(action, 2, turn_delta_sigma=2.0, turn_damping=0.9,
                    weapon_switch_prob=0.2-0.17*self.epsilon)
            elif r < self.epsilon*4.0:
                action = mutate_action(action, 1, turn_delta_sigma=1.5, turn_damping=0.85,
                    weapon_switch_prob=0.1-0.08*self.epsilon)

        # Add some random walk to epsilon
        # self.epsilon += np.random.normal(scale=1.0/256)
        # self.epsilon = np.clip(self.epsilon, 0.01, 1.0)

        return action

    def pick_top_replay_entries(self):
        #return self.memory.get_best_clutch(256) + list(self.memory.get_best_entries(128))
        return self.memory.sequence

    def mix_reward(self, reward_model, reward_game, reward_system):
        return 5.0*reward_model + reward_game + reward_system
        #return reward_action