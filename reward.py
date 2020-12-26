import vizdoom as vzd

#import matplotlib.pyplot as plt
import numpy as np

"""
Hardcoded start pos of first level
"""
def get_start_pos():
    return np.array((-96.0, 784.78125, 56.0))

"""
ret: (x,y,z)
"""
def get_player_pos(game):
    return np.array((game.get_game_variable(vzd.POSITION_X),
    game.get_game_variable(vzd.POSITION_Y),
    game.get_game_variable(vzd.POSITION_Z)))

def print_state_sectors(game):
    state = game.get_state()
    for s in state.sectors:
        print("Sector lines:", [(l.x1, l.y1, l.x2, l.y2, l.is_blocking) for l in s.lines])

def print_state_objects(game):
    state = game.get_state()
    for o in state.objects:
        print("Object id:", o.id, "object name:", o.name)
        print("Object position: x:", o.position_x, ", y:", o.position_y, ", z:", o.position_z)

def get_player_dist_from_start(game):
    player = get_player_pos(game)
    start_point = get_start_pos()
    return np.linalg.norm(player-start_point)

def get_reward(game):
    startdist_reward = get_player_dist_from_start(game)

    return startdist_reward