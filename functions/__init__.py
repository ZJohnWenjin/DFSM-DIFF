import torch.optim as optim


def get_optimizer(config, parameters):
    if config.optim_Diff.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim_Diff.lr, weight_decay=config.optim_Diff.weight_decay,
                          betas=(config.optim_Diff.beta1, 0.999), amsgrad=config.optim_Diff.amsgrad,
                          eps=config.optim_Diff.eps)
    elif config.optim_Diff.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim_Diff.lr, weight_decay=config.optim_Diff.weight_decay)
    elif config.optim_Diff.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim_Diff.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim_Diff.optimizer))
