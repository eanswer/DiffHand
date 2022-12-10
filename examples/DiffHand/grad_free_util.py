import nevergrad as ng
import numpy as np

def optimize_params(optim_name, loss_func, num_params, init_values, max_iters, num_workers=1, bounds=None, popsize=None):
    parametrization = ng.p.Array(init=init_values)

    if bounds is not None:
        parametrization.set_bounds(lower=bounds[:, 0], upper=bounds[:, 1])
    if 'CMA' in optim_name:
        optim_func = getattr(ng.optimizers, '_CMA')
    else:
        optim_func = getattr(ng.optimizers, optim_name)
    optim_dict = dict(
        parametrization=parametrization, budget=max_iters, num_workers=num_workers,
    )
    if 'CMA' in optim_name and popsize is not None:
        print(f'Setting popsize to be:{popsize}')
        optim_dict['popsize'] = popsize
    if optim_name == 'FCMA':
        optim_dict['fcmaes'] = True
    elif optim_name == 'DiagonalCMA':
        optim_dict['diagonal'] = True
    optimizer = optim_func(**optim_dict)

    losses = []
    losses.append(np.array([0, loss_func(init_values)]))
    if num_workers > 1:
        raise ValueError('not working yet!')
    else:
        for i in range(optimizer.budget):
            x = optimizer.ask()
            loss = loss_func(x.value)
            print(f'iteration {i}: loss {loss}')
            optimizer.tell(x, loss)
            losses.append(np.array([i + 1, loss]))
    recommendation = optimizer.provide_recommendation()
    params = recommendation.value
    return params, np.array(losses)