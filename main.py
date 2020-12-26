#!/usr/bin/env python3

#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function
import vizdoom as vzd
import numpy as np

from random import choice
from time import sleep

from init_game import init_game
from reward import Reward
from model import Model
import utils

def main():
    game = init_game()
    episodes = 10

    # Sets time that will pause the engine after each action (in seconds)
    # Without this everything would go too fast for you to keep track of what's happening.
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    game.new_episode()

    player_start_pos = utils.get_player_pos(game)
    print("Player start pos:", player_start_pos)

    reward_controller = Reward(player_start_pos)
    model = Model(reward_controller)

    for i in range(episodes):
        #if i % 50 == 0:
        #    print("Episode #" + str(i + 1))

        game.new_episode()
        rewards = []

        while not game.is_episode_finished():
            state = game.get_state()
            state_number = state.number
            screen_buf = state.screen_buffer

            reward = model.step(game)

            rewards.append(reward)

            #reward = game.make_action(model.predict_action(np.expand_dims(screen_buf,0)))

            # state, reward, done = env.step(action)
            # remember cur state, action, reward, new state, done
            # replay
            # iterate target model
            
            if state_number % 50 == 0:
                print("State #" + str(state_number))
                print("Reward:", reward)
                #print("=====================")
                print()

            # disable sleep unless a human wants to watch the game
            #if sleep_time > 0:
            #    sleep(sleep_time)

        
        # Check how the episode went.
        print("Episode", i, "finished in", )
        #print("Total reward:", game.get_total_reward())
        print("Total reward:", sum(rewards))
        print("************************")

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()

print()
print("-------- starting ------------")
main()