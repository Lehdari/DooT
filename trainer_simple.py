from trainer_interface import *
from utils import *
import random
import math


class TrainerSimple(TrainerInterface):
    def episode_reset(self, model):
        TrainerInterface.episode_reset(self, model)

        self.epsilon = 1.0/(1.0 + math.exp((self.episode_id - 256.0)/48.0))
        
        self.epsilon = 1.0
        # self.epsilon = 0.0

    def pick_action(self, game, model):
        # action = np.array([-1.0+2.0*random.random() for i in range(15)])
        # action = np.concatenate([np.zeros((14,), dtype=np.float32), np.array([-1.0+2.0*random.random()])])
        # return action

        r = random.random()
        if r < self.epsilon:
            if random.random() < 0.05:
                action = get_random_action(turn_delta_sigma=5.0,
                    weapon_switch_prob=0.03-0.026*self.epsilon)
            else:
                action = mutate_action(self.action_prev, 2, turn_delta_sigma=5.0, turn_damping=0.9,
                    weapon_switch_prob=0.03-0.026*self.epsilon)
            action[14] = 0.95*self.action_prev[14] + 0.05*action[14]
        else:
            action = model.predict_action(self.memory.active_episode)

            if r < self.epsilon*2.0:
                action = mutate_action(action, 2, turn_delta_sigma=3.0, turn_damping=0.9,
                    weapon_switch_prob=0.02-0.017*self.epsilon)
            elif r < self.epsilon*4.0:
                action = mutate_action(action, 1, turn_delta_sigma=2.0, turn_damping=0.95,
                    weapon_switch_prob=0.01-0.008*self.epsilon)

        # Add some random walk to epsilon
        # self.epsilon += np.random.normal(scale=1.0/128)
        # self.epsilon = np.clip(self.epsilon, 0.0, 1.0)

        # TEMP: no attack for now
        action[0] = -0.9

        return action

    def pick_top_replay_entries(self):
        return self.memory.sequence

    def mix_reward(self, reward_model, reward_game, reward_system):
        return reward_model + reward_game + reward_system