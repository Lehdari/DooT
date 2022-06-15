#!/usr/bin/env python3

import vizdoom as vzd
import numpy as np
import argparse
import concurrent.futures

from init_game import init_game
from reward import Reward
from model import Model
from trainer_simple import TrainerSimple
import utils

import faulthandler
faulthandler.enable()


def main():
    parser = argparse.ArgumentParser()
    model_filename = ""
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    model_filename = args.model
    model_filename = "model" # TODO TEMP

    runs = 16384
    episode_length = 2048
    min_episode_length = 1024
    replay_sample_length = 256
    # episode_length = 128
    # min_episode_length = 128
    # replay_sample_length = 96
    n_replay_episodes = 8
    n_training_epochs = 8
    window_visible = False
    output_visual_log = False

    # game.new_episode()

    # player_start_pos = utils.get_player_pos(game)
    # print("Player start pos:", player_start_pos)

    reward_controller = Reward()
    model = Model(episode_length, n_replay_episodes,
        n_training_epochs, replay_sample_length,
        output_visual_log)

    if model_filename is not None:
        model.load_model(model_filename, model_filename)
    trainer = TrainerSimple(reward_controller, n_replay_episodes, episode_length,
        min_episode_length, window_visible)

    print("Model setup complete. Starting training episodes")

    memory = trainer.run(model)
    for i in range(runs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # start a new memory gathering run concurrently
            memory_future = executor.submit(trainer.run, model.create_copy())
        
            model.train(memory)

            # replace memory with the new one
            del memory
            memory = memory_future.result()

print()
print("-------- starting ------------")
main()
sys.exit(0)
