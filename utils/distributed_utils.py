from torch.utils.data import Sampler
import math
import os
import pdb
import torch
import torch.distributed as dist
from torch.nn import Module
import multiprocessing as mp
import numpy as np


class ParallelWrapper(Module):
    def __init__(self, net, parallel_mode='none'):
        super(ParallelWrapper, self).__init__()
        assert parallel_mode in ['dist', 'data_parallel', 'none']
        self.parallel_mode = parallel_mode
        if parallel_mode == 'none':
            self.net = net
            self.module = net
        elif parallel_mode == 'dist':
            self.net = DistModule(net)
            self.module = self.net.module
        else:
            self.net = torch.nn.DataParallel(net)
            self.module = self.net.module

    def forward(self, *inputs, **kwargs):
        return self.net.forward(*inputs, **kwargs)

    def train(self, mode=True):
        super(ParallelWrapper, self).train(mode)
        self.net.train(mode)


class DistModule(Module):
    def __init__(self, module):
        super(DistModule, self).__init__()
        self.module = module
        broadcast_params(self.module)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def train(self, mode=True):
        super(DistModule, self).train(mode)
        self.module.train(mode)

def gradients_multiply(model, multiplier=1):
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            param.grad.data *= multiplier
    
def average_gradients(model):
    """ average gradients """

    # for n, param, in model.named_parameters():
    #     if 'dynamic_sigma' in n:
    #         print(param.requires_grad, param.grad.data, param.data)
    #     if param.requires_grad and param.grad is not None:
    #         dist.all_reduce(param.grad.data)

    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad.data)
            # param.grad.data *= multiplier


def broadcast_params(model):
    """ broadcast model parameters """
    for p in model.state_dict().values():
        dist.broadcast(p, 0)


def dist_init(port):
    # os.environ["OMP_NUM_THREADS"] = "1"
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)

    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    #added by dy
    # addr = node_list[8:].replace('-', '.')
    addr = node_list.replace('-', '.')
    if ',' in addr:
        addr = addr.split(',')[0]
    addr = addr[8:]
    # addr = ','.join([ad[8:] for ad in addrs ])

    print(addr)

    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


# from . import Sampler


class DistributedSequatialSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        # g = torch.Generator()
        # g.manual_seed(self.epoch)
        # indices = torch.randperm(len(self.dataset), generator=g).tolist()

        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedGivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1):
        if world_size is None:
            world_size = dist.get_world_size()  # link.get_world_size()
        if rank is None:
            rank = dist.get_rank()  # link.get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter*self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter+1)*self.batch_size:])
        else:
            return iter(self.indices[(self.last_iter+1)*self.batch_size:])
            raise RuntimeError(
                "this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        # each process shuffle all list with same seed, and pick one piece according to rank
        # np.random.seed(0)
        np.random.seed(7)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size-1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg:beg+self.total_size]

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        # return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size

    def set_epoch(self, epoch):
        pass


class DistributedGivenIterationSamplerEpoch(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1, review_cycle=-1):
        if world_size is None:
            world_size = dist.get_world_size()  # link.get_world_size()
        if rank is None:
            rank = dist.get_rank()  # link.get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter*self.batch_size
        self.review_cycle = review_cycle # in unit of epoch

        self.indices = self.gen_new_list()
        self.call = 0
    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter+1)*self.batch_size:])
        else:
            return iter(self.indices[(self.last_iter+1)*self.batch_size:])
            # raise RuntimeError(
            #     "this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        # each process shuffle all list with same seed, and pick one piece according to rank
        # np.random.seed(0)
        np.random.seed(7)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size-1) // indices.shape[0] + 1

        # indices = np.tile(indices, num_repeat)
        indices = np.concatenate([np.random.permutation(indices) for i in range(num_repeat) ] )
        seeds = np.arange(indices.size).reshape(indices.shape)

        if self.review_cycle>0:
            assert (1/self.review_cycle)%1==0
            # review_freq = 1/1/self.review_cycle
            h = len(indices) // int(self.review_cycle*len(self.dataset))
            
            # print(indices.shape,'???!!!',  indices[:h*int(self.review_cycle*len(self.dataset) ) ].shape)
            indices = indices[:h*int(self.review_cycle*len(self.dataset) ) ].reshape([h,-1] )
            seeds = seeds[:h*int(self.review_cycle*len(self.dataset) ) ].reshape([h,-1] )

            indices = np.concatenate([indices, indices], axis=1).reshape(-1)
            seeds = np.concatenate([seeds, seeds], axis=1).reshape(-1)

        indices = indices[:all_size]
        seeds = seeds[:all_size]

        # np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg:beg+self.total_size]
        seeds = seeds[beg:beg+self.total_size]

        assert len(indices) == self.total_size

        # return indices
        return list(zip(list(indices), list(seeds)))

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        # return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size

    def set_epoch(self, epoch):
        pass
