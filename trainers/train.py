from __future__ import division
import torch
import torch.optim as optim
import models
from setproctitle import setproctitle as ptitle
from worlds.atari import atari_env
from misc.util import ensure_shared_grads
from misc.player_util import Agent
from torch.autograd import Variable


def train(rank, config, shared_model, optimizer, env_conf):
    ptitle('Training Agent: {}'.format(rank))

    # Set GPU
    gpu_id = config.trainer.gpu_ids[rank % len(config.trainer.gpu_ids)]
    torch.manual_seed(config.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(config.seed + rank)

    # Set env
    env = atari_env(config.game.env, env_conf, config)
    env.seed(config.seed + rank)

    # Set optimizer if not specified
    if optimizer is None:
        if config.trainer.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(
                shared_model.parameters(),
                lr=config.trainer.lr)
        if config.trainer.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(),
                lr=config.trainer.lr,
                amsgrad=config.trainer.amsgrad)

    # Set player
    player = Agent(None, env, config, None)
    player.gpu_id = gpu_id
    player.model = models.load(
        config,
        player.env.observation_space.shape[0],
        player.env.action_space)
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    player.model.train()
    player.eps_len += 2

    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())

        # Set LSTM state
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = Variable(torch.zeros(1, 512).cuda())
                    player.hx = Variable(torch.zeros(1, 512).cuda())
            else:
                player.cx = Variable(torch.zeros(1, 512))
                player.hx = Variable(torch.zeros(1, 512))
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)

        # In the paper: "The policy and the value function are updated every
        # tmmax actions or when a terminal state is reached"
        for step in range(config.algorithm.n_steps):
            player.action_train()
            if player.done:
                break

        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _ = player.model(
                (Variable(player.state.unsqueeze(0)), (player.hx, player.cx)))
            R = value.data  # For non-terminal s_t, Bootstrap from last state

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        for i in reversed(range(len(player.rewards))):
            R = config.algorithm.gamma * R + player.rewards[i]

            # Set value loss (critic loss)
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = \
                player.rewards[i] + \
                config.algorithm.gamma * player.values[i + 1].data - \
                player.values[i].data

            gae = gae * config.algorithm.gamma * config.algorithm.tau + delta_t

            # Set policy loss (actor loss)
            policy_loss = \
                policy_loss - \
                player.log_probs[i] * Variable(gae) - \
                0.01 * player.entropies[i]

        # Update network
        player.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()
