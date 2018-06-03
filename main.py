from __future__ import print_function, division
import os
# os.environ["OMP_NUM_THREADS"] = "1"
import torch
import yaml
import time
from environment import atari_env
from utils import read_config
from model import A3Clstm
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
from misc.util import Struct


def configure():
    with open("config.yaml") as config_f:
        config = Struct(**yaml.load(config_f))

    return config

if __name__ == '__main__':
    config = configure()
    torch.manual_seed(config.seed)
    if config.trainer.gpu_ids == -1:
        config.trainer.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(config.seed)
        mp.set_start_method('spawn')
    setup_json = read_config(config.game.crop_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in config.game.env:
            env_conf = setup_json[i]
    env = atari_env(config.game.env, env_conf, config)
    import torch.multiprocessing as mp
    shared_model = A3Clstm(env.observation_space.shape[0], env.action_space)
    if config.load:
        saved_state = torch.load(
            '{0}{1}.dat'.format(config.load_model_dir, config.env),
            map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    if config.trainer.shared_optimizer:
        if config.trainer.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=config.trainer.lr)
        if config.trainer.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=config.trainer.lr, amsgrad=config.trainer.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []

    p = mp.Process(target=test, args=(config, shared_model, env_conf))
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
