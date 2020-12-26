import vizdoom as vzd
import numpy as np
import utils

class Reward():
    def __init__(self, player_start_pos):
        self.player_start_pos = player_start_pos
    
    def get_sector_crossing_reward(self):
        return 0.0 # TODO

    def get_reward(self, game):
        start_dist_reward = utils.get_player_dist_from_start(game, self.player_start_pos)
        living_reward = -1.0

        sector_crossing_reward = self.get_sector_crossing_reward() # TODO


        return start_dist_reward + living_reward + sector_crossing_reward