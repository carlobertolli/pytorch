#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.
# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=./common-build.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

#export ROCM_PATH=/root/workdir/rocm-asan-711-0119



echo "Clang version:"
$ROCM_PATH/llvm/bin/clang --version

# hipify sources
python tools/amd_build/build_amd.py

# sccache somehow forces gfx906 -x hip, remove it all
#pushd $ROCM_PATH/llvm/bin
#if [[ -d original ]]; then
#  mv original/clang .
#  mv original/clang++ .
#fi
#rm -rf original
#popd
#rm -rf /opt/cache

# patch XNNPACK to work around build failure
pushd third_party/XNNPACK
patch -p1 -i ../../.ci/pytorch/XNNPACK.patch || true
popd

python tools/stats/export_test_times.py

# shellcheck source=./env-rocm-asan.sh
export BUILD_ONLY_ENV_VARS=1
source "$(dirname "${BASH_SOURCE[0]}")/env-therock-asan.sh"

echo $PWD
#. .ci/pytorch/build.sh

echo $PYTORCH_ROCM_ARCH
VERBOSE=1 python setup.py develop 2>&1 | tee build.log
#python setup.py bdist_wheel
#pip_install_whl "$(echo dist/*.whl)"

# Local build test
#$ROCM_PATH/bin/hipcc /my_home/tests/memory_leak/test.cpp -o test

