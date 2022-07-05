#!/usr/bin/env python3

import vizdoom as vzd
import numpy as np
import argparse
import concurrent.futures
import gc

from init_game import init_game
from memory import Memory
from reward import Reward
from model import Model
from trainer_simple import TrainerSimple
import utils

import faulthandler
faulthandler.enable()

from os import listdir, mkdir, path
from os.path import isfile, join, isdir
import sys
import datetime

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help="model name in directory models/", default="model")
    parser.add_argument('--quiet', action="store_true", help="suppress logging and plots", default=False)
    parser.add_argument('--use-concurrent-training', action="store_true",
        help="train and collect gameplay data concurrently", default=False)
    parser.add_argument('--runs', type=int, help="number of training runs", default=16384)
    parser.add_argument('--episode-length', type=int,
        help="how many frames or steps the episode can be at maximum before it is ended",
        default=4096)
    parser.add_argument('--min-episode-length', type=int,
        help="minimum episode length. episodes shorter than this are discarded",
        default=2048)
    parser.add_argument('--replay-sample-length', type=int,
        help="how many frames are sampled from the memory each training epoch",
        default=128)
    parser.add_argument('--smoketest-length', type=int,
        help="length of sequences in overfitting smoketest",
        default=32)
    parser.add_argument('--n-replay-episodes', type=int, help="",
        default=8)
    parser.add_argument('--n-training-epochs', type=int,
        help="number of training epochs per each training run",
        default=8)
    parser.add_argument('--window-visible', action="store_true",
        help="show vizdoom window during training (this is different from opencv/matplotlib plot images)",
        default=False)

    # Currently not used probably
    parser.add_argument('--output-visual-log', action="store_true",
        help="collect and save image flats, image encs, image enc preds and states to out/", default=False)

    args = parser.parse_args()
    if args.min_episode_length > args.episode_length:
        args.min_episode_length = args.episode_length
    if args.replay_sample_length > args.episode_length:
        args.replay_sample_length = args.episode_length
    if not args.use_concurrent_training:
        if args.replay_sample_length > args.smoketest_length:
            args.replay_sample_length = args.smoketest_length

    return args

def main(args):
    model_filename = args.model
    episode_length = args.episode_length
    runs = args.runs
    min_episode_length = args.min_episode_length # TEMP Eljas: the piece of code that uses this is commented away
    replay_sample_length = args.replay_sample_length
    smoketest_length = args.smoketest_length
    n_replay_episodes = args.n_replay_episodes # This has something to do with the action model. currently it's not used.
    n_training_epochs = args.n_training_epochs
    window_visible = args.window_visible
    output_visual_log = args.output_visual_log
    quiet = args.quiet
    use_concurrent_training = args.use_concurrent_training

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_output_dir = 'logs/' + current_time + '/train'

    reward_controller = Reward()
    model = Model(episode_length, n_replay_episodes,
        n_training_epochs, replay_sample_length,
        model_output_dir,
        output_visual_log, quiet)

    if model_filename is not None:
        model_dir = "model"
        if isdir(model_dir):
            print(f"Found directory {model_dir}")
            print("Searching saved neural network models")
            h5files = [f for f in listdir(model_filename) if isfile(join(model_filename, f))]
            h5files = [f for f in h5files if ".h5" in f]
            h5files = [f.split("_")[0] for f in h5files]
            h5files = [f for f in h5files if "-" in f]
            if h5files:
                model.load_model(model_filename, h5files[0])
            else:
                print(f"Model with name {model_filename} not found in {model_dir}, creating new model")
        else:
            print(f"Did not find directory {model_dir}. Creating the directory and creating a new model")
            mkdir(model_dir)
    else:
        print("Model filename not specified. Exiting.")
        return
    
    trainer = TrainerSimple(reward_controller, n_replay_episodes, episode_length,
        min_episode_length, window_visible, replay_sample_length)

    print("Model setup complete. Starting training episodes")


    if not use_concurrent_training:
        smoketest_memory_filename = f"data/smoketest/memory_{smoketest_length}.npz"
        if path.exists(smoketest_memory_filename):
            print(f"Loading {smoketest_memory_filename}")
            memory = Memory(n_replay_episodes, smoketest_length)
            memory.load(smoketest_memory_filename)
        else:
            print(f"{smoketest_memory_filename} not found, creating new smoketest memory")
            memory = trainer.run(model, is_smoketest=True, smoketest_length=smoketest_length)
            print(f"Saving {smoketest_memory_filename}")
            memory.save(smoketest_memory_filename)
        
        for i in range(runs):
            print(f"Run {i} / {runs}")
            model.train(memory)
    else:
        for i in range(runs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # start a new memory gathering run concurrently
                memory_future = executor.submit(trainer.run, model.create_copy())
            
                model.train(memory)

                # replace memory with the new one
                del memory
                gc.collect()
                memory = memory_future.result()


if __name__=="__main__":
    args=parse_args()

    print()
    print("-------- starting ------------")
    main(args)
    sys.exit(0)
