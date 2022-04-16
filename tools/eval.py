#CERTIFICATED
import torch
import numpy as np 
import tensorboard
from pathlib import Path
import json
import random
import re
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import time
import fire
import torch.distributed as dist
import os
from collections import defaultdict
import ast 
import flow_vis
import copy

from utils.progress_bar import ProgressBar
from utils.log_tool import SimpleModelLog
from data.preprocess import merge_batch, get_dataloader, get_dataloader_deepim # merge_second_batch_multigpu
from utils.config_io import merge_cfg, save_cfg
import torchplus
from builder import (
    dataset_builder,
    input_reader_builder,
    lr_scheduler_builder,
    optimizer_builder,
    rnnpose_builder
)
from utils.distributed_utils import dist_init, average_gradients, DistModule, ParallelWrapper, DistributedSequatialSampler, DistributedGivenIterationSampler, DistributedGivenIterationSamplerEpoch 
from utils.util import modify_parameter_name_with_map
from utils.eval_metric import *
from config.default import get_cfg



GLOBAL_GPUS_PER_DEVICE = 1  
GLOBAL_STEP = 0
RANK=-1
WORLD_SIZE=-1


def load_example_to_device(example,
                             device=None) -> dict:
    example_torch = {}

    for k, v in example.items():  
        if k in ['idx', 'class_name']:
            example_torch[k]=v
            continue

        if type(v) == list:
            example_torch[k] = [item.to(device=device) for item in v]
        else:
            example_torch[k] = v.to(device=device)

    return example_torch
def build_network(model_cfg, measure_time=False, testing=False):
    net = rnnpose_builder.build(
        model_cfg, measure_time=measure_time, testing=testing)
    return net


def _worker_init_fn(worker_id):
    global GLOBAL_STEP
    time_seed = GLOBAL_STEP
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])


def freeze_params(params: dict, include: str = None, exclude: str = None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    remain_params = []
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                continue
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                continue
        remain_params.append(p)
    return remain_params


def freeze_params_v2(params: dict, include: str = None, exclude: str = None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                p.requires_grad = False
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                p.requires_grad = False


def filter_param_dict(state_dict: dict, include: str = None, exclude: str = None):
    assert isinstance(state_dict, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    res_dict = {}
    for k, p in state_dict.items():
        if include_re is not None:
            if include_re.match(k) is None:
                continue
        if exclude_re is not None:
            if exclude_re.match(k) is not None:
                continue
        res_dict[k] = p
    return res_dict

def chk_rank(rank_, use_dist=False):
    if not use_dist:
        return True
    global RANK
    if RANK<0:
        RANK=dist.get_rank()
    cur_rank = RANK#dist.get_rank()
    # self.world_size = dist.get_world_size()
    return cur_rank == rank_

def get_rank(use_dist=False):
    if not use_dist:
        return 0
    else:
        # return dist.get_rank()
        global RANK 
        if RANK<0:
            RANK=dist.get_rank()
        return RANK 

def get_world(use_dist):
    if not use_dist:
        return 1
    else:
        global WORLD_SIZE 
        if WORLD_SIZE<0:
            WORLD_SIZE=dist.get_world_size()
        return WORLD_SIZE #dist.get_world_size()
def get_ngpus_per_node():
    global GLOBAL_GPUS_PER_DEVICE
    return GLOBAL_GPUS_PER_DEVICE


def multi_proc_train(
          config_path,
          model_dir,
          use_apex,
          world_size,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          pretrained_path=None,
          pretrained_include=None,
          pretrained_exclude=None,
          pretrained_param_map=None,
          freeze_include=None,
          freeze_exclude=None,
          measure_time=False,
          resume=False,
          use_dist=False,
          gpus_per_node=1,
          start_gpu_id=0,
          optim_eval=False,
          seed=7,
          dist_port="23335",
         force_resume_step=None,
         batch_size=None,
         apex_opt_level='O0'
          ):
    
    params = {
          "config_path": config_path,
          "model_dir": model_dir,
          "use_apex": use_apex,
          "result_path": result_path,
          "create_folder": create_folder,
          "display_step": display_step,
          "summary_step": summary_step,
          "pretrained_path": pretrained_path,
          "pretrained_include": pretrained_include,
          "pretrained_exclude": pretrained_exclude,
          "pretrained_param_map": pretrained_param_map,
          "freeze_include": freeze_include,
          "freeze_exclude": freeze_exclude,
        #   "multi_gpu": multi_gpu,
          "measure_time": measure_time,
          "resume": resume,
          "use_dist": use_dist,
          "gpus_per_node": gpus_per_node,
          "optim_eval": optim_eval,
          "seed": seed,
          "dist_port": dist_port,
          "world_size": world_size,
          "force_resume_step":force_resume_step,
          "batch_size": batch_size,
          "apex_opt_level":apex_opt_level
    }
    from types import SimpleNamespace 
    params = SimpleNamespace(**params)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(x) for x in range(start_gpu_id, start_gpu_id+gpus_per_node))
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"  )

    mp.spawn(train_worker, nprocs=gpus_per_node,
                args=( params,) )

def train_worker(rank, params):
    global RANK, WORLD_SIZE
    RANK = rank
    WORLD_SIZE=params.world_size
    
    eval(config_path=params.config_path,
          model_dir=params.model_dir,
          use_apex=params.use_apex,
          result_path=params.result_path,
          create_folder=params.create_folder,
          display_step=params.display_step,
          pretrained_path=params.pretrained_path,
          pretrained_include=params.pretrained_include,
          pretrained_exclude=params.pretrained_exclude,
          pretrained_param_map=params.pretrained_param_map,
          freeze_include=params.freeze_include,
          freeze_exclude=params.freeze_exclude,
          measure_time=params.measure_time,
          resume=params.resume,
          use_dist=params.use_dist,
          dist_port=params.dist_port,
          gpus_per_node=params.gpus_per_node,
          optim_eval=params.optim_eval,
          seed=params.seed,
          force_resume_step=params.force_resume_step,
          batch_size = params.batch_size,
          apex_opt_level=params.apex_opt_level
          ) 


def eval(
         config_path,
          model_dir,
          use_apex,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          pretrained_path=None,
          pretrained_include=None,
          pretrained_exclude=None,
          pretrained_param_map=None,
          freeze_include=None,
          freeze_exclude=None,
          multi_gpu=False,
          measure_time=False,
          resume=False,
          use_dist=False,
          dist_port="23335",
          gpus_per_node=1,
          optim_eval=False,
          seed=7,
          force_resume_step=None,
          batch_size=None,
          apex_opt_level='O0',
          verbose=False
          ):
    """train a VoxelNet model specified by a config file.
    """

    print("force_resume_step:", force_resume_step)
    print("torch.cuda.is_available()=", torch.cuda.is_available())
    print("torch.version.cuda=",torch.version.cuda) 
    dist_url=f"tcp://127.0.0.1:{dist_port}"
    print(f"dist_url={dist_url}", flush=True)
    global RANK, WORLD_SIZE
    # RANK, WORLD_SIZE=rank, world_size
    if RANK<0:
        RANK=0
    if WORLD_SIZE<0:
        WORLD_SIZE=1

    global GLOBAL_GPUS_PER_DEVICE
    GLOBAL_GPUS_PER_DEVICE = gpus_per_node

  

    ######################################## initialize the distributed env #########################################
    if use_dist:
        if use_apex:
            dist.init_process_group(
                backend="nccl", init_method=dist_url, world_size=get_world(use_dist), rank=get_rank(use_dist))
        else:
            # rank, world_size = dist_init(str(dist_port))
            dist.init_process_group(
                backend="nccl", init_method=dist_url, world_size=get_world(use_dist), rank=get_rank(use_dist))
    
    print(get_rank(use_dist)%GLOBAL_GPUS_PER_DEVICE, flush=True)
    #set cuda device number
    torch.cuda.set_device(get_rank(use_dist)%GLOBAL_GPUS_PER_DEVICE)

    ############################################ create folders ############################################
    print(f"Set seed={seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model_dir = str(Path(model_dir).resolve())
    model_dir = Path(model_dir)
    if chk_rank(0, use_dist):
        if not resume and model_dir.exists():
            raise ValueError("model dir exists and you don't specify resume.")
            print("Warning: model dir exists and you don't specify resume.")

        model_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"

    ############################################# read config proto ############################################
    config = merge_cfg(
        [config_path], intersection=True)
    if chk_rank(0, use_dist):
        print(json.dumps(config, indent=4))

    if chk_rank(0, use_dist):
        # save_cfg([default_config_path, custom_config_path],
        save_cfg([config_path, config_path],
                 str(model_dir / config_file_bkp))

    #update the global config object
    get_cfg().merge(config.get("BASIC",{}),"BASIC" )  

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model
    train_cfg = config.train_config
    optimizer_cfg = train_cfg.optimizer
    loss_scale = train_cfg.loss_scale_factor


    ############################################# Update default options ############################################

    if batch_size is not None:
        input_cfg.batch_size = batch_size 
        eval_input_cfg.batch_size = batch_size
    print(input_cfg.batch_size)
    
    ############################################# build network, optimizer etc. ############################################
    #dummy dataset
    dataset_tmp = input_reader_builder.build(
        eval_input_cfg,
        training=False,
    )
 
    model_cfg.obj_seqs = copy.copy(dataset_tmp._dataset.infos["seqs"])

    net = build_network(model_cfg, measure_time)  # .to(device)
    net.cuda()

    fastai_optimizer = optimizer_builder.build(
        optimizer_cfg, net,
        mixed=False,
        loss_scale=loss_scale)

    print("# parameters:", len(list(net.parameters())))

    ############################################# load pretrained model ############################################
    if pretrained_path is not None:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')

        if verbose:
            print("Pretrained keys:", pretrained_dict.keys())
            print("Model keys:", model_dict.keys())

        pretrained_dict = filter_param_dict(
            pretrained_dict, pretrained_include, pretrained_exclude)

        pretrained_dict = modify_parameter_name_with_map(
            pretrained_dict, ast.literal_eval(str(pretrained_param_map)))
        new_pretrained_dict = {}
        
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                new_pretrained_dict[k] = v
            else:
                print("Fail to load:", k )

        model_dict.update(new_pretrained_dict)
        net.load_state_dict(model_dict)
        freeze_params_v2(dict(net.named_parameters()),
                         freeze_include, freeze_exclude)
        net.clear_global_step()
        # net.clear_metrics()
        del pretrained_dict
    else:
    ############################################# try to resume from the latest chkpt ############################################
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    # torchplus.train.try_restore_latest_checkpoints(model_dir,
    #                                                [fastai_optimizer])

    ######################################## parallel the network  #########################################
    if use_dist:
        if use_apex:
            import apex
            net, amp_optimizer = apex.amp.initialize(net.cuda(
            ), fastai_optimizer, opt_level="O0", keep_batchnorm_fp32=None, loss_scale=None)
            net_parallel = apex.parallel.DistributedDataParallel(net)
        else:
            # net_parallel = ParallelWrapper(net.cuda(), 'dist')
            # amp_optimizer = fastai_optimizer
            amp_optimizer = optimizer_builder.build(
                optimizer_cfg, net,
                mixed=False,
                loss_scale=loss_scale)
            net_parallel = torch.nn.parallel.DistributedDataParallel(net, device_ids=[get_rank(use_dist)], output_device=get_rank(use_dist) ,find_unused_parameters=True)
    else:
        net_parallel = net.cuda()

    ############################################# build lr_scheduler ############################################
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, amp_optimizer,
                                              train_cfg.steps)

    float_dtype = torch.float32
    ######################################## build dataloaders #########################################
    if use_dist:
        num_gpu = 1
        collate_fn = merge_batch
    else:
        raise NotImplementedError

    print(f"MULTI-GPU: using {num_gpu} GPU(s)")

        
    ######################
    # PREPARE INPUT
    ######################
    dataset = input_reader_builder.build(
        input_cfg,
        training=True,
    )
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        training=False,
    )


    if use_dist:
        train_sampler = DistributedGivenIterationSamplerEpoch(
            dataset, train_cfg.steps, input_cfg.batch_size, last_iter=net.get_global_step()-1, review_cycle=-1)
        # train_sampler=DistributedSequatialSampler(dataset)
        shuffle = False
        eval_sampler = DistributedSequatialSampler(eval_dataset)

    else:
        train_sampler = None
        eval_sampler = None
        eval_train_sampler = None
        shuffle = True

    dataloader, neighborhood_limits=get_dataloader_deepim(dataset=dataset,
                                                kpconv_config=model_cfg.descriptor_net.keypoints_detector_3d,
                                                batch_size=input_cfg.batch_size * num_gpu,
                                                shuffle=shuffle,
                                                num_workers= 2 ,#input_cfg.preprocess.num_workers * num_gpu,
                                                sampler=train_sampler 
                                        )
    eval_dataloader, _ =get_dataloader_deepim(dataset=eval_dataset,
                                                kpconv_config=model_cfg.descriptor_net.keypoints_detector_3d,
                                    batch_size=eval_input_cfg.batch_size,
                                    shuffle=False,
                                    num_workers= 2, #eval_input_cfg.preprocess.num_workers,
                                    sampler=eval_sampler,
                                    neighborhood_limits=neighborhood_limits
                                   )

    #########################################################################################
    #                                            TRAINING
    ##########################################################################################
    model_logging = SimpleModelLog(model_dir, disable=get_rank(use_dist) != 0)
    model_logging.open()
    start_step = net.get_global_step()
    total_step = train_cfg.steps
    t = time.time()
    steps_per_eval = train_cfg.steps_per_eval

    amp_optimizer.zero_grad()
    step = start_step
    epoch = 0
    net_parallel.eval()

    classes=list(set(eval_dataset.dataset.infos['seqs']))
    
    evaluator = dict([(c,LineMODEvaluator(f"{c}",result_path) ) for c in classes ] )

    ang_errs=[]
    trans_errs=[]
    try:
        for example in eval_dataloader:
            global GLOBAL_STEP
            GLOBAL_STEP = step


            lr_scheduler.step(net.get_global_step())
            example_torch=load_example_to_device(example, device=torch.device("cuda"))

            batch_size = example["image"].shape[0]
            with torch.no_grad():
                # t1=time.time()
                ret_dict = net_parallel(example_torch)
                # print("time:", time.time()-t1)

            eval_results=evaluator[example['class_name'][0]].evaluate_rnnpose(ret_dict, example_torch )
            ang_errs.append(eval_results['ang_err'])
            trans_errs.append(eval_results['trans_err'])

            if step%10 ==0:
                model_logging.log_metrics({
                    "ang_err": float(ang_errs[-1]), 
                    "trans_err": float(trans_errs[-1]), 
                }, GLOBAL_STEP)

                model_logging.log_images(
                    {
                        "pc_proj_vis": eval_results['pc_proj_vis'].transpose([2,0,1])[None], # HWC->NCHW
                        "pc_proj_vis_pred": eval_results['pc_proj_vis_pred'].transpose([2,0,1])[None], # HWC->NCHW
                        "syn_img": torch.cat(ret_dict["syn_img"], dim=0).detach().cpu(),
                        # "image": example["image"].cpu(),
                        # "image_f": (example["image"].cpu()+ret_dict["syn_img"][0].detach().cpu())/2,
                    }, GLOBAL_STEP, prefix="")




            net.update_global_step()
            metrics = defaultdict(dict)
            GLOBAL_STEP = net.get_global_step()

            step += 1


        for k in evaluator:
            print(f"###############Evaluation results of class {k}###############")
            evaluator[k].summarize()
                
    except Exception as e:
        model_logging.log_text(str(e), step)
        raise e
    finally:
        model_logging.close()



if __name__ == '__main__':
    fire.Fire()
