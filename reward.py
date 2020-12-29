import vizdoom as vzd
import numpy as np
from utils import *

class Reward():
    def __init__(self, player_start_pos):
        self.player_start_pos = player_start_pos
        self.dist_start_prev = 0.0
    
    # reset the reward system state (after an episode)
    def reset(self):
        self.dist_start_prev = 0.0
    
    def get_sector_crossing_reward(self):
        return 0.0 # TODO

    def get_reward(self, game):
    	# current distance from starting point
        dist_start = np.linalg.norm(get_player_pos(game) - self.player_start_pos)
        # starting distance reward given according to dis. delta
        start_dist_reward = dist_start - self.dist_start_prev
        self.dist_start_prev = dist_start

        living_reward = -1.0

        sector_crossing_reward = self.get_sector_crossing_reward() # TODO


        return start_dist_reward + living_reward + sector_crossing_reward