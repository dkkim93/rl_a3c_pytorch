from __future__ import division
import torch
import time
import logging
import models
from setproctitle import setproctitle as ptitle
from worlds.atari import atari_env
from misc.util import setup_logger
from misc.player_util import Agent
from torch.autograd import Variable


def test(config, shared_model, env_conf):
    ptitle('Test Agent')

    # Set log
    log = {}
    setup_logger('{}_log'.format(config.game.env), r'{0}{1}_log'.format(
        config.log_dir, config.game.env))
    log['{}_log'.format(config.game.env)] = logging.getLogger('{}_log'.format(
        config.game.env))
    d_args = vars(config)
    for k in d_args.keys():
        log['{}_log'.format(config.game.env)].info('{0}: {1}'.format(k, d_args[k]))

    # Set GPU
    torch.manual_seed(config.seed)
    gpu_id = config.trainer.gpu_ids[-1]
    if gpu_id >= 0:
        torch.cuda.manual_seed(config.seed)

    # Set env
    env = atari_env(config.game.env, env_conf, config)

    # Set player
    player = Agent(None, env, config, None)
    player.gpu_id = gpu_id
    player.model = models.load(
        config,
        player.env.observation_space.shape[0],
        player.env.action_space)
    player.state = player.env.reset()
    player.eps_len += 2
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()

    flag = True
    max_score = 0
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    while True:
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            player.model.eval()  # NOTE Set to eval for test (needs to be set only once)
            flag = False

        # Take one step action
        player.action_test()
        reward_sum += player.reward

        if player.done and not player.info:
            state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        # If env's "was_real_done" flag is True, then reset
        elif player.info:
            flag = True
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(config.game.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    reward_sum,
                    player.eps_len,
                    reward_mean))

            if config.trainer.save_max and reward_sum >= max_score:
                max_score = reward_sum
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}.dat'.format(
                            config.save_model_dir, config.game.env))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}.dat'.format(
                        config.save_model_dir, config.game.env))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.eps_len += 2
            time.sleep(10)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
