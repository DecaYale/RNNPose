import numpy as np 
import torch

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
import kornia
import flow_vis
import copy
import ast 

from utils.progress_bar import ProgressBar
from utils.log_tool import SimpleModelLog
from data.preprocess import merge_batch, get_dataloader_deepim  # merge_second_batch_multigpu
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
# from utils.visualize import vis_pointclouds_cv2, vis_2d_keypoints_cv2
from utils.util import modify_parameter_name_with_map
from config.default import get_cfg
from data.ycb.basic import bop_ycb_class2idx


GLOBAL_GPUS_PER_DEVICE = 1  # None
GLOBAL_STEP = 0
RANK=-1
WORLD_SIZE=-1


def load_example_to_device(example,
                             device=None) -> dict:
    # global GLOBAL_GPUS_PER_DEVICE
    # device = device % GLOBAL_GPUS_PER_DEVICE or torch.device("cuda:0")
    # example_torch = defaultdict(list)

    example_torch = {}

    for k, v in example.items():  
        if k in ['class_name', 'idx'] or example[k] is None:
            example_torch[k] = v
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
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger



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
          "apex_opt_level":apex_opt_level,
          "start_gpu_id": start_gpu_id
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

    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(params.start_gpu_id+rank) 
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    torch.cuda.set_device(rank%params.gpus_per_node)
    
    train(config_path=params.config_path,
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
        #   multi_gpu=params.multi_gpu,
          measure_time=params.measure_time,
          resume=params.resume,
          use_dist=params.use_dist,
          dist_port=params.dist_port,
          gpus_per_node=params.gpus_per_node,
          optim_eval=params.optim_eval,
          seed=params.seed,
          force_resume_step=params.force_resume_step,
          batch_size = params.batch_size,
          apex_opt_level=params.apex_opt_level,
          gpu_id=params.start_gpu_id+rank, 
          ) 


def train(
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
          gpu_id=None
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

    #fix the seeds 
    print(f"Set seed={seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


  
    ######################################## initialize the distributed env #########################################
    if use_dist:
        # torch.cuda.set_device(get_rank(use_dist))
        if use_apex:
            dist.init_process_group(
                backend="nccl", init_method=dist_url, world_size=get_world(use_dist), rank=get_rank(use_dist))
        else:
            # rank, world_size = dist_init(str(dist_port))
            dist.init_process_group(
                backend="nccl", init_method=dist_url, world_size=get_world(use_dist), rank=get_rank(use_dist))
    

    ############################################ create folders ############################################

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
    print(input_cfg.batch_size)
    
    ############################################# build network, optimizer etc. ############################################
    #dummy dataset to get obj_seqs
    dataset_tmp = input_reader_builder.build(
        input_cfg,
        training=True,
    )
 
    model_cfg.obj_seqs = copy.copy(dataset_tmp._dataset.infos["seqs"])
    print(model_cfg.obj_seqs, type(model_cfg.obj_seqs), flush=True)

    model_cfg.gpu_id=gpu_id
    net = build_network(model_cfg, measure_time)  #.to(device)
    net.cuda()

    fastai_optimizer = optimizer_builder.build(
        optimizer_cfg, net,
        mixed=False,
        loss_scale=loss_scale)

    print("num parameters:", len(list(net.parameters())))

    ############################################# load pretrained model ############################################
    if pretrained_path is not None:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
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
        print("Load pretrained parameters:")
        for k, v in new_pretrained_dict.items():
            print(k, v.shape)
        model_dict.update(new_pretrained_dict)
        net.load_state_dict(model_dict)
        freeze_params_v2(dict(net.named_parameters()),
                         freeze_include, freeze_exclude)
        net.clear_global_step()
        # net.clear_metrics()
        del pretrained_dict
    ############################################# try to resume from the latest chkpt ############################################
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [fastai_optimizer])

    ######################################## parallel the network  #########################################
    if use_dist:
        if use_apex:
            import apex
            net = apex.parallel.convert_syncbn_model(net)
            net, amp_optimizer = apex.amp.initialize(net.cuda(
            ), fastai_optimizer, opt_level="O0", keep_batchnorm_fp32=None, loss_scale=None)
            net_parallel = apex.parallel.DistributedDataParallel(net)
        else:
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

    if 0:  # TODO:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32
    ######################################## build dataloaders #########################################
    if use_dist:
        num_gpu = 1
        collate_fn = merge_batch
    else:
        raise NotImplementedError
    print(f"MULTI-GPU: use {num_gpu} gpu")

        
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
                                                num_workers=input_cfg.preprocess.num_workers * num_gpu,
                                                sampler=train_sampler 
                                        )
    eval_dataloader, _ =get_dataloader_deepim(dataset=eval_dataset,
                                                kpconv_config=model_cfg.descriptor_net.keypoints_detector_3d,
                                    batch_size=eval_input_cfg.batch_size,
                                    shuffle=False,
                                    num_workers=eval_input_cfg.preprocess.num_workers,
                                    sampler=eval_sampler,
                                    neighborhood_limits=neighborhood_limits
                                   )

    ##########################################################################################
    #                                            TRAINING
    ##########################################################################################
    model_logging = SimpleModelLog(model_dir, disable=get_rank(use_dist) != 0)
    model_logging.open()
    start_step = net.get_global_step()
    total_step = train_cfg.steps
    t = time.time()
    steps_per_eval = train_cfg.steps_per_eval

    amp_optimizer.zero_grad()
    step_times = []
    step = start_step
    epoch = 0
    net_parallel.train()

    try:
        while True:

            if use_dist:
                epoch = (net.get_global_step() *
                         input_cfg.batch_size) // len(dataloader)

                dataloader.sampler.set_epoch(epoch)
            else:
                epoch += 1
            for example in dataloader:

                global GLOBAL_STEP
                GLOBAL_STEP = step

                lr_scheduler.step(net.get_global_step())
                example_torch=load_example_to_device(example, device=torch.device("cuda"))

                batch_size = example["image"].shape[0]

                ret_dict = net_parallel(example_torch)

                loss = ret_dict["loss"].mean()  # /get_world(use_dist)
                recall = ret_dict['recall'].mean()

                reduced_loss = loss.data.clone() / get_world(use_dist)
                reduced_recall = recall.data.clone() / get_world(use_dist)
                if use_dist:
                    dist.all_reduce_multigpu(
                        [reduced_loss])
                    dist.all_reduce_multigpu(
                        [reduced_recall])

                amp_optimizer.zero_grad()
                if use_apex:
                    with apex.amp.scale_loss(loss, amp_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss = loss/get_world(use_dist)
                    loss.backward()
                    if use_dist:
                        average_gradients(net_parallel)


                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                amp_optimizer.step()

                net.update_global_step()
                step_time = (time.time() - t)
                step_times.append(step_time)
                t = time.time()
                metrics = defaultdict(dict)
                GLOBAL_STEP = net.get_global_step()

                if chk_rank(0, use_dist) and GLOBAL_STEP % display_step == 0:
                    print(f'Model directory: {str(model_dir)}')
                    if measure_time:
                        for name, val in net.get_avg_time_dict().items():
                            print(f"avg {name} time = {val * 1000:.3f} ms")

                    metrics["runtime"] = {
                        "step": GLOBAL_STEP,
                        "steptime": np.mean(step_times),
                    }

                    metrics["loss"]["loss"] = float(
                        reduced_loss.detach().cpu().numpy())
                    # metrics["loss"]["translation_loss"] = float(
                    metrics["recall"] = float(reduced_recall.detach().cpu().numpy())

                    metrics['learning_rate'] = amp_optimizer.lr
                    metrics['reproj_loss'] = float(ret_dict['reproj_loss'].median().detach().cpu().numpy())
                    metrics['loss_3d_proj'] = float(ret_dict['loss_3d_proj'].median().detach().cpu().numpy())
                    # metrics['chamfer_loss'] = float(ret_dict['chamfer_loss'].median().detach().cpu().numpy())
                    if hasattr(net.motion_net, "sigma"):
                        metrics['sigma'] = float(net.motion_net.sigma[0].detach().cpu().numpy())

                    metrics['epoch'] = epoch

                    model_logging.log_metrics(metrics, GLOBAL_STEP)


                    
                    if isinstance(ret_dict["flow"], (list, tuple) ):
                        ret_dict["flow"] = ret_dict["flow"][-1]
                    flow=flow_vis.flow_to_color(ret_dict["flow"][0].squeeze().permute(1,2,0).detach().cpu().numpy(), convert_to_bgr=False)

                    model_logging.log_images(
                    {
                        "image": example["image"].cpu(),
                        "flow": flow.transpose(2,0,1)[None],
                        "weight": ret_dict["weight"].squeeze(1).mean(1,keepdims=True).detach().cpu(),
                        "syn_img": torch.cat(ret_dict["syn_img"], dim=0)[:,:3].detach().cpu(),
                        "syn_depth": (ret_dict["syn_depth"][-1]/ret_dict["syn_depth"][-1].max()).detach().cpu(),
                        "valid_mask": ret_dict["valid_mask"].detach().squeeze(1).permute(0,3,1,2).cpu(),
                        "ren_mask": example["ren_mask"][:,None].cpu()
                    }, GLOBAL_STEP, prefix="") 

                if optim_eval and GLOBAL_STEP < total_step/2 and GLOBAL_STEP > train_cfg.steps_per_eval*2:
                    steps_per_eval = 2*train_cfg.steps_per_eval
                else:
                    steps_per_eval = train_cfg.steps_per_eval

                if GLOBAL_STEP % steps_per_eval == 0:
                    if chk_rank(0, use_dist):  # logging
                        torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                                    net.get_global_step())
                    eval_once(net,
                              eval_dataset=eval_dataset, eval_dataloader=eval_dataloader, eval_input_cfg=eval_input_cfg,
                              result_path=result_path,
                              global_step=GLOBAL_STEP,
                              model_logging=model_logging,
                              metrics=metrics,
                              float_dtype=float_dtype,
                              use_dist=use_dist,
                              prefix='eval_')

                    net.train()

                step += 1
                if step >= total_step:
                    break
            if step >= total_step:
                break
    except Exception as e:
        model_logging.log_text(str(e), step)
        raise e
    finally:
        model_logging.close()
    torchplus.train.save_models_cpu(model_dir, [net, amp_optimizer],
                                    net.get_global_step())





def eval_once(net,
              eval_dataset, eval_dataloader, eval_input_cfg,
              result_path, global_step, model_logging, metrics, float_dtype, use_dist, prefix='eval_'):
    from utils.eval_metric import LineMODEvaluator#, YCBEvaluator

    net.eval()
    result_path_step = result_path / \
        f"step_{global_step}"
    if chk_rank(0, use_dist):
        result_path_step.mkdir(parents=True, exist_ok=True)
        model_logging.log_text("#################################",
                               global_step)
        model_logging.log_text("# EVAL", global_step)
        model_logging.log_text("#################################",
                               global_step)
        model_logging.log_text(
            "Generate output labels...", global_step)
        prog_bar = ProgressBar()
        prog_bar.start(len(eval_dataloader))

    # RES={
    #     # "recall":[]
    #     "recall":[],
    #     "3d_proj_error":[],
    #     "2d_proj_error":[],
    # }
    t = 0
    detections = []
    
    #construct evaluator
    classes=list(set(eval_dataset.dataset.infos['seqs']))
    if classes[0] in bop_ycb_class2idx.keys():
        evaluator = dict([(c,YCBEvaluator(f"{c}",result_path) ) for c in classes ] )
    else:
        evaluator = dict([(c,LineMODEvaluator(f"{c}",result_path) ) for c in classes ] )

    # cnt = 0  
    for i, example in enumerate(eval_dataloader):
        example_torch=load_example_to_device(example, device=torch.device("cuda"))
        batch_size = example["image"].shape[0]
        device = example_torch["image"].device
        with torch.no_grad():
            ret_dict = net(example_torch)

        # RES['recall'].append(recall.detach())
        # RES['3d_proj_error'].append(ret_dict['loss_3d_proj'].mean().detach() )
        # RES['2d_proj_error'].append(ret_dict['reproj_loss'].mean().detach() )

        #do evaluation
        evaluator[example['class_name'][0]].evaluate_rnnpose(ret_dict, example)


        if chk_rank(0, use_dist) and i % 10 == 0:
            prog_bar.print_bar(finished_size=10)
    eval_res={} 
    for k in evaluator:
        eval_res[k]=evaluator[k].summarize()

    if use_dist:  # chk_rank(0, use_dist):
        #gather and summarize evaluation results
        for k in eval_res:
            eval_res[k]['seq_len'] =  torch.tensor(eval_res[k]['seq_len'], device=device)
            gather_list = [torch.zeros_like(eval_res[k]['seq_len'])
                           for i in range(get_world(use_dist))]
            dist.all_gather(gather_list, eval_res[k]['seq_len'])
            seq_len = torch.stack(gather_list, dim=-1).sum()

            for kk in eval_res[k]:
                if kk =='seq_len':
                    continue
                eval_res[k][kk] =  torch.tensor(eval_res[k][kk], device=device)
                eval_res[k][kk] =  eval_res[k][kk]*eval_res[k]['seq_len'] # calculate the sum 
                gather_list = [torch.zeros_like(eval_res[k][kk] )
                           for i in range(get_world(use_dist))]
                dist.all_gather(gather_list, eval_res[k][kk])
                eval_res[k][kk] = torch.stack(gather_list, dim=-1).sum()/seq_len #divide with the whole length to get the mean value 



    if chk_rank(0, use_dist):
        for k in eval_res:
            for kk in eval_res[k]:
                if kk == 'seq_len':
                    metrics[f'{k}_{kk}'] = float(seq_len.cpu().numpy() )
                metrics[f'{k}_{kk}'] = float(eval_res[k][kk].cpu().numpy())


    if chk_rank(0, use_dist):
        model_logging.log_metrics(metrics, global_step)
    # del RES 
    del eval_res
    net.train()


if __name__ == '__main__':
    fire.Fire()
