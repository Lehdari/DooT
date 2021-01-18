from trainer_interface import *
from utils import *
import random


class TrainerSimple(TrainerInterface):
    def pick_action(self, game):
        if self.episode_id % 4 == 3:
            return self.model.predict_worst_action()
        else:
            return self.model.predict_action()
        
        # if self.episode_id < 128:
        #     return get_random_action(turn_delta_sigma=5.0)
        # if self.episode_id < 2048:
        #     if random.random() < 0.25:
        #         if random.random() < 0.4:
        #             return mutate_action(self.model.predict_action(), 2, weapon_switch_prob=0.06)
        #         else:
        #             return self.model.predict_action()
        #     else:
        #         return mutate_action(self.action_prev, 4, weapon_switch_prob=0.1)
        # elif self.episode_id < 4096:
        #     if random.random() < 0.35:
        #         if random.random() < 0.5:
        #             return mutate_action(self.model.predict_action(), 1, weapon_switch_prob=0.05)
        #         else:
        #             return self.model.predict_action()
        #     else:
        #         return mutate_action(self.action_prev, 3, weapon_switch_prob=0.1)
        # else:
        #     if random.random() < 0.45:
        #         if random.random() < 0.6:
        #             return mutate_action(self.model.predict_action(), 1, weapon_switch_prob=0.05)
        #         else:
        #             return self.model.predict_action()
        #     else:
        #         return mutate_action(self.action_prev, 3, weapon_switch_prob=0.1)

    def pick_top_replay_entries(self):
        # return self.memory.get_best_clutch(256) + list(self.memory.get_best_entries(128))
        return self.memory.sequence

    def mix_reward(self, reward_model, reward_game, reward_system, reward_action):
            return reward_model*20.0 + reward_game + reward_system + reward_action