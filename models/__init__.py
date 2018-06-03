from a3c_lstm import A3Clstm


def load(config, obs_space, action_space):
    model_name = config.algorithm.model
    try:
        cls = globals()[model_name]
        #return cls(env.observation_space.shape[0], env.action_space)
        return cls(obs_space, action_space)
    except KeyError:
        raise Exception("No such model: {}".format(model_name))
