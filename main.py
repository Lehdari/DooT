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
from trainer_simple import TrainerSimple
import utils
import argparse

import faulthandler
faulthandler.enable()


def main():
    parser = argparse.ArgumentParser()
    model_filename = ""
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    model_filename = args.model
    model_filename = "model/model" # TODO TEMP

    runs = 16384
    episode_length = 4096
    replay_sample_length = 512
    n_replay_episodes = 8
    n_training_epochs = 8
    game = init_game(episode_length)

    game.new_episode()

    player_start_pos = utils.get_player_pos(game)
    print("Player start pos:", player_start_pos)

    reward_controller = Reward(player_start_pos)
    model = Model(episode_length, n_replay_episodes, n_training_epochs, replay_sample_length)

    if model_filename is not None:
        print("Loading model ({})".format(model_filename))
        model.load_model(model_filename)
    trainer = TrainerSimple(model, reward_controller, n_replay_episodes, episode_length,
        2*replay_sample_length)

    print("Model setup complete. Starting training episodes")

    for i in range(runs):
        memory = trainer.run(game)
        model.train(memory)

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()

print()
print("-------- starting ------------")
main()
sys.exit(0)
