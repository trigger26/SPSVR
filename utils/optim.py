import numpy as np
import math

from torch import optim


def create_optimizer(config, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = config.optim.lower()
    weight_decay = config.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_args = dict(lr=config.lr, weight_decay=weight_decay)
    if 'opt_eps' in config and config.opt_eps is not None:
        opt_args['eps'] = config.opt_eps
    if 'opt_betas' in config and config.opt_betas is not None:
        opt_args['betas'] = config.opt_betas

    print("optimizer settings:", opt_args)

    if opt_lower == 'sgd':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=config.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=config.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    
    else:
        raise ValueError("Invalid optimizer")

    return optimizer


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())


def warmup_scheduler(base_value, niter_per_ep, warmup_epochs=0, start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    return warmup_schedule, warmup_iters


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule, warmup_iters = warmup_scheduler(base_value, niter_per_ep, warmup_epochs, start_warmup_value, warmup_steps)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def exponential_scheduler(base_value, gamma, epochs, niter_per_ep, warmup_epochs=0,
                          start_warmup_value=0, warmup_steps=-1):
    warmup_schedule, warmup_iters = warmup_scheduler(base_value, niter_per_ep, warmup_epochs, start_warmup_value, warmup_steps)

    schedule = np.ones(niter_per_ep) * base_value
    schedule = np.concatenate((warmup_schedule, schedule))
    
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([base_value * (gamma ** (i // niter_per_ep)) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule
