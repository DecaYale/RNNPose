#!/usr/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_DIR/../thirdparty/kpconv/cpp_wrappers
bash compile_wrappers.sh

cd $SCRIPT_DIR/../thirdparty/nn
python setup.py build_ext --inplace