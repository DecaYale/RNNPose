
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJ_ROOT=$SCRIPT_DIR/../
export PYTHONPATH=$PROJ_ROOT:PYTHONPATH

mkdir --parent $PROJ_ROOT/EXPDATA/data_info/deepim/

python $PROJ_ROOT/tools/generate_data_info_deepim_0_orig.py  create_data_info  --data_root $PROJ_ROOT/EXPDATA/LM6d_converted/LM6d_refine  --saving_path $PROJ_ROOT/EXPDATA/data_info/deepim/linemod_orig_deepim.info  

python $PROJ_ROOT/tools/generate_data_info_deepim_1_syn.py  create_data_info  --data_root $PROJ_ROOT/EXPDATA/LM6d_converted/LM6d_refine_syn  --saving_path $PROJ_ROOT/EXPDATA/data_info/deepim/linemod_syn_deepim.info  --with_assertion True

python $PROJ_ROOT/tools/generate_data_info_deepim_2_posecnnval.py  create_data_info  --data_root $PROJ_ROOT/EXPDATA/LM6d_converted/LM6d_refine  --saving_path $PROJ_ROOT/EXPDATA/data_info/linemod_posecnn.info  --with_assertion True

python $PROJ_ROOT/tools/generate_data_info_v2_deepim.py  create_data_info  --data_root $PROJ_ROOT/EXPDATA/LINEMOD/fuse_formatted/  --saving_path $PROJ_ROOT/EXPDATA/data_info/linemod_fusesformatted_all10k_deepim.info  --training_data_ratio 1 --shuffle=False
