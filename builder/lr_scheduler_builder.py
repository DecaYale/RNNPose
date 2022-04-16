
from torchplus.train import learning_schedules_fastai as lsf
import torch
import numpy as np 

def build(optimizer_config, optimizer, total_step):

    optimizer_type = list(optimizer_config.keys())[0]

    if optimizer_type == 'rms_prop_optimizer':
        config = optimizer_config.rms_prop_optimizer
        lr_scheduler = _create_learning_rate_scheduler(
            config.learning_rate, optimizer, total_step=total_step)

    if optimizer_type == 'momentum_optimizer':
        config = optimizer_config.momentum_optimizer
        lr_scheduler = _create_learning_rate_scheduler(
            config.learning_rate, optimizer, total_step=total_step)

    if optimizer_type == 'adam_optimizer':
        config = optimizer_config.adam_optimizer
        lr_scheduler = _create_learning_rate_scheduler(
            config.learning_rate, optimizer, total_step=total_step)

    return lr_scheduler


def _create_learning_rate_scheduler(learning_rate_config, optimizer, total_step):
    """Create optimizer learning rate scheduler based on config.

    Args:
      learning_rate_config: A LearningRate proto message.

    Returns:
      A learning rate.

    Raises:
      ValueError: when using an unsupported input data type.
    """
    lr_scheduler = None
    # learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
    learning_rate_type = list(learning_rate_config.keys())[0]

    if learning_rate_type == 'multi_phase':
        config = learning_rate_config.multi_phase
        lr_phases = []
        mom_phases = []
        for phase_cfg in config.phases:
            lr_phases.append((phase_cfg.start, phase_cfg.lambda_func))
            mom_phases.append(
                (phase_cfg.start, phase_cfg.momentum_lambda_func))
        lr_scheduler = lsf.LRSchedulerStep(
            optimizer, total_step, lr_phases, mom_phases)



    if learning_rate_type == 'one_cycle':
        config = learning_rate_config.one_cycle

        if len(config.lr_maxs)>1:
          assert(len(config.lr_maxs)==4 )    
          lr_max=[]
          # for i in range(len(config.lr_maxs)):
          #   lr_max += [config.lr_maxs[i]]*optimizer.param_segs[i] 

          lr_max = np.array(list(config.lr_maxs) )
        else:
          lr_max = config.lr_max

        lr_scheduler = lsf.OneCycle(
            optimizer, total_step, lr_max, list(config.moms), config.div_factor, config.pct_start)
    if learning_rate_type == 'exponential_decay':
        config = learning_rate_config.exponential_decay
        lr_scheduler = lsf.ExponentialDecay(
            optimizer, total_step, config.initial_learning_rate, config.decay_length, config.decay_factor, config.staircase)
    if learning_rate_type == 'exponential_decay_warmup':
        config = learning_rate_config.exponential_decay_warmup
        lr_scheduler = lsf.ExponentialDecayWarmup(
            optimizer, total_step, config.initial_learning_rate, config.decay_length, config.decay_factor,   config.div_factor,
            config.pct_start, config.staircase)
    if learning_rate_type == 'manual_stepping':
        config = learning_rate_config.manual_stepping
        lr_scheduler = lsf.ManualStepping(
            optimizer, total_step, list(config.boundaries), list(config.rates))

    if lr_scheduler is None:
        raise ValueError('Learning_rate %s not supported.' %
                         learning_rate_type)

    return lr_scheduler
