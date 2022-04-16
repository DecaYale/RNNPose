SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJ_ROOT=$SCRIPT_DIR/../

python $PROJ_ROOT/tools/transform_data_format.py run --data_type "LM_FUSE_PVNET"  --data_info_path $PROJ_ROOT/EXPDATA/"/data_info/linemod_all_10k_default.info.all"  --image_root $PROJ_ROOT/EXPDATA/raw_data/fuse --depth_root $PROJ_ROOT/EXPDATA/raw_data/orig_renders --save_dir $PROJ_ROOT/EXPDATA/LINEMOD/fuse_formatted  && touch 3.finished 


