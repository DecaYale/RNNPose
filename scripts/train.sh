export PROJECT_ROOT_PATH=/home/RNNPose/Projects/Works/RNNPose_release

export PYTHONPATH="$PROJECT_ROOT_PATH:$PYTHONPATH"
export PYTHONPATH="$PROJECT_ROOT_PATH/thirdparty:$PYTHONPATH"
export model_dir='outputs'
seq=cat
gpu=1
start_gpu_id=0
mkdir $model_dir

train_file=$PROJECT_ROOT_PATH/tools/train.py
config_path=/$PROJECT_ROOT_PATH/config/linemod/"$seq"_fw0.5.yml
# pretrain=$PROJECT_ROOT_PATH/weights/trained_models/"$seq".tckpt

python -u $train_file multi_proc_train  \
        --config_path $config_path \
        --model_dir $model_dir/results \
        --use_dist True \
        --dist_port 10000 \
        --gpus_per_node $gpu \
        --optim_eval True \
        --use_apex False \
        --world_size $gpu \
        --start_gpu_id $start_gpu_id \
        # --pretrained_path $pretrain 
 