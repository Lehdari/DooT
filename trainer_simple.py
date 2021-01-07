from trainer_interface import *
from utils import *
import random


class TrainerSimple(TrainerInterface):
    def pick_action(self, game):
        if self.episode_id < 64:
            return get_random_action(turn_delta_sigma=5.0)
        else:
            if random.random() < 0.6:
                if random.random() < 0.6:
                    return mutate_action(self.model.predict_action(), 1, weapon_switch_prob=0.05)
                else:
                    return self.model.predict_action()
            else:
                return mutate_action(self.action_prev, 3, weapon_switch_prob=0.1)

    def mix_reward(self, reward_model, reward_game, reward_system):
            return reward_model*50.0 + reward_game + reward_system