import vizdoom as vzd
import numpy as np
import collections
from utils import *


class Reward():
    def __init__(self):
        # self.player_start_pos = player_start_pos

        # exploration
        self.exploration_tile_size = 64.0
        self.exploration_decay_rate = 0.05
        self.exploration_tiles = {}

        self.reset()

    
    # reset the reward system state (after an episode)
    def reset(self):
        self.dist_start_prev = 0.0

        # items
        self.weapon0_prev = -1
        self.weapon1_prev = -1
        self.weapon2_prev = -1
        self.weapon3_prev = -1
        self.weapon4_prev = -1
        self.weapon5_prev = -1
        self.weapon6_prev = -1
        self.ammo2_prev = -1
        self.ammo3_prev = -1
        self.ammo5_prev = -1
        self.ammo6_prev = -1
        
        # combat
        self.health_prev = -1.0
        self.armor_prev = -1.0
        self.damage_prev = -1.0

        self.velocity = 0.0

        # buffer for storing recent turn deltas
        self.turn_buffer_size = 128
        self.turn_buffer = collections.deque(maxlen=self.turn_buffer_size)
    
    def reset_exploration(self):
        self.exploration_tiles = {}
    
    def get_velocity_reward(self, game):
        vx = game.get_game_variable(vzd.VELOCITY_X)
        vy = game.get_game_variable(vzd.VELOCITY_Y)
        # some low pass filter to smooth out jitter
        self.velocity = 0.9 * self.velocity + 0.1 * np.sqrt(vx*vx + vy*vy)
        return self.velocity/8.33 - 1.0 # error -1 when standing still, 0 for walking, 1 for running
    
    def get_item_reward(self, game):
        weapon0 = game.get_game_variable(vzd.WEAPON0)
        weapon1 = game.get_game_variable(vzd.WEAPON1)
        weapon2 = game.get_game_variable(vzd.WEAPON2)
        weapon3 = game.get_game_variable(vzd.WEAPON3)
        weapon4 = game.get_game_variable(vzd.WEAPON4)
        weapon5 = game.get_game_variable(vzd.WEAPON5)
        weapon6 = game.get_game_variable(vzd.WEAPON6)
        ammo2 = game.get_game_variable(vzd.AMMO2)
        ammo3 = game.get_game_variable(vzd.AMMO3)
        ammo5 = game.get_game_variable(vzd.AMMO5)
        ammo6 = game.get_game_variable(vzd.AMMO6)

        if self.weapon0_prev < 0:
            self.weapon0_prev = weapon0
        if self.weapon1_prev < 0:
            self.weapon1_prev = weapon1
        if self.weapon2_prev < 0:
            self.weapon2_prev = weapon2
        if self.weapon3_prev < 0:
            self.weapon3_prev = weapon3
        if self.weapon4_prev < 0:
            self.weapon4_prev = weapon4
        if self.weapon5_prev < 0:
            self.weapon5_prev = weapon5
        if self.weapon6_prev < 0:
            self.weapon6_prev = weapon6
        if self.ammo2_prev < 0:
            self.ammo2_prev = ammo2
        if self.ammo3_prev < 0:
            self.ammo3_prev = ammo3
        if self.ammo5_prev < 0:
            self.ammo5_prev = ammo5
        if self.ammo6_prev < 0:
            self.ammo6_prev = ammo6
        
        weapon_reward = weapon0 - self.weapon0_prev
        weapon_reward += weapon1 - self.weapon1_prev
        weapon_reward += weapon2 - self.weapon2_prev
        weapon_reward += weapon3 - self.weapon3_prev
        weapon_reward += weapon4 - self.weapon4_prev
        weapon_reward += weapon5 - self.weapon5_prev
        weapon_reward += weapon6 - self.weapon6_prev

        ammo_reward = (ammo2 - self.ammo2_prev) * 5.0 # bullets
        ammo_reward += (ammo3 - self.ammo3_prev) * 20.0 # shells
        ammo_reward += (ammo5 - self.ammo5_prev) * 50.0 # rockets
        ammo_reward += (ammo6 - self.ammo6_prev) * 12.0 # plasma
        
        self.ammo2_prev = ammo2
        self.ammo3_prev = ammo3
        self.ammo5_prev = ammo5
        self.ammo6_prev = ammo6

        self.weapon0_prev = weapon0
        self.weapon1_prev = weapon1
        self.weapon2_prev = weapon2
        self.weapon3_prev = weapon3
        self.weapon4_prev = weapon4
        self.weapon5_prev = weapon5
        self.weapon6_prev = weapon6

        return ammo_reward * 0.5 + weapon_reward * 1000.0

    def get_combat_reward(self, game):
        health = game.get_game_variable(vzd.HEALTH)
        armor = game.get_game_variable(vzd.ARMOR)
        damage = game.get_game_variable(vzd.DAMAGECOUNT)

        if self.health_prev < 0:
            self.health_prev = health
        if self.armor_prev < 0:
            self.armor_prev = armor
        if self.damage_prev < 0:
            self.damage_prev = damage
        
        damage_reward = damage - self.damage_prev
        health_reward = health - self.health_prev
        armor_reward = armor - self.armor_prev

        # if damage_reward != 0:
        #     print("damage_reward: {}".format(damage_reward))
        # if health_reward != 0:
        #     print("health_reward: {}".format(health_reward))
        # if armor_reward != 0:
        #     print("armor_reward: {}".format(armor_reward))
        
        self.damage_prev = damage
        self.health_prev = health
        self.armor_prev = armor

        return 20.0*damage_reward + health_reward + armor_reward

    def get_exploration_reward(self, player_pos):
        tile_x = int(player_pos[0] / self.exploration_tile_size)
        tile_y = int(player_pos[1] / self.exploration_tile_size)
        tile_id = (tile_x, tile_y)
        if tile_id in self.exploration_tiles:
            # tile-wise reward decay
            self.exploration_tiles[tile_id] = self.exploration_tiles[tile_id]*\
                (1.0-self.exploration_decay_rate) - self.exploration_decay_rate
        else:
            # initialize tile reward according to distance to starting point
            tile_x_middle = (tile_id[0] + 0.5)*self.exploration_tile_size
            tile_y_middle = (tile_id[1] + 0.5)*self.exploration_tile_size
            tile_dist_start = np.linalg.norm(np.array([tile_x_middle, tile_y_middle]) -\
                self.player_start_pos[0:2])

            print(f"{self.player_start_pos}")
            print(f"{tile_dist_start}")
            
            self.exploration_tiles[tile_id] = 1.0 +\
                np.power(max(tile_dist_start/2.0-30.0, 0.0), 0.3)
            
            #self.exploration_tiles[tile_id] = self.exploration_tile_init

        
        self.exploration_tiles[tile_id]

        # print("x: {:8.3f} y: {:8.3f} tile: {} value: {}".format(
        #      player_pos[0], player_pos[1], tile_id, self.exploration_tiles[tile_id]))
        
        return self.exploration_tiles[tile_id]
    
    def get_start_distance_reward(self, player_pos):
    	# current distance from starting point
        dist_start = np.linalg.norm(player_pos - self.player_start_pos)
        # starting distance reward given according to dis. delta
        start_dist_reward = dist_start - self.dist_start_prev
        self.dist_start_prev = dist_start
        return start_dist_reward

    def get_action_reward(self, action):
        reward_action = 0.0

        # # by default, penalize from weapon change actions
        # for i in range(7, 14):
        #     if action[i]:
        #         reward_action -= 1.0

        # # also penalize for pressing forward+back or right+left simultaneously
        # if action[3] and action[4]:
        #     reward_action -= 1.0
        # if action[5] and action[6]:
        #     reward_action -= 1.0
        
        # # encourage attacking
        # if action[0]:
        #     reward_action += 1.0

        # penalize for "spinbottiness":
        # heavy penalties for continuous rotation, (hence the turn delta buffering)
        self.turn_buffer.append(action[14])
        turn_delta_buffered = sum(self.turn_buffer)/self.turn_buffer_size
        reward_action -= 8.0*abs(turn_delta_buffered)**3.0

        return reward_action

    def get_misc_reward(self, game):
        return game.get_game_variable(vzd.ATTACK_READY) - 1.0

    def get_reward(self, game, action):
        player_pos = get_player_pos(game)

        living_reward = 0.0

        velocity_reward = self.get_velocity_reward(game)

        #start_dist_reward = 0.0#self.get_start_distance_reward(player_pos)

        #exploration_reward = 0.0#self.get_exploration_reward(player_pos)

        item_reward = self.get_item_reward(game)

        combat_reward = self.get_combat_reward(game)
        
        action_reward = self.get_action_reward(action)

        misc_reward = self.get_misc_reward(game)
        #print(misc_reward)

        return\
            living_reward +\
            1.0*velocity_reward +\
            0.5*item_reward +\
            2.0*combat_reward +\
            1.0*action_reward +\
            1.0*misc_reward
    
    def get_distance(self, game):
        return np.linalg.norm(get_player_pos(game) - self.player_start_pos)
