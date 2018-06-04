from __future__ import print_function, division
import torch
import torch.multiprocessing as mp
import yaml
import time
import models
from worlds.atari import atari_env
from trainers.train import train
from trainers.test import test
from trainers.shared_optim import SharedRMSprop, SharedAdam
from misc.util import Struct, read_config


def configure():
    with open("config.yaml") as config_f:
        config = Struct(**yaml.load(config_f))

    return config


def setup_env(config):
    setup_json = read_config(config.game.crop_config)
    env_conf = setup_json["Default"]  # TODO May not be needed as overrided below
    for game in setup_json.keys():
        if game in config.game.env:
            env_conf = setup_json[game]
    env = atari_env(config.game.env, env_conf, config)

    return env, env_conf


def setup_model(env, config):
    shared_model = models.load(
        config, env.observation_space.shape[0], env.action_space)
    if config.load:
        saved_state = torch.load(
            '{0}{1}.dat'.format(config.load_model_dir, config.env),
            map_location=lambda storage,
            loc: storage)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()  # NOTE Hogwild style

    return shared_model


def setup_optimizer(shared_model, config):
    if config.trainer.shared_optimizer:
        if config.trainer.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(
                shared_model.parameters(),
                lr=config.trainer.lr)
        if config.trainer.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(),
                lr=config.trainer.lr,
                amsgrad=config.trainer.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    return optimizer


def train_model(shared_model, env_conf, optimizer, config):
    processes = []

    p = mp.Process(
        target=test, args=(config, shared_model, env_conf))
    p.start()
    processes.append(p)
    time.sleep(0.1)

    for rank in range(0, config.trainer.n_workers):
        p = mp.Process(
            target=train, args=(rank, config, shared_model, optimizer, env_conf))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()


if __name__ == '__main__':
    config = configure()
    torch.manual_seed(config.seed)
    if config.trainer.gpu_ids == -1:
        config.trainer.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(config.seed)
        mp.set_start_method('spawn')

    # Set env
    env, env_conf = setup_env(config)

    # Set model
    shared_model = setup_model(env, config)

    # Set optimizer
    optimizer = setup_optimizer(shared_model, config)

    # Train
    train_model(shared_model, env_conf, optimizer, config)
