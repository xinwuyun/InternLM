#!/bin/bash
set -x

[[ -n ${GITHUB_WORKSPACE} ]] || { echo "should set GITHUB_WORKSPACE first before ci, exit."; exit 1; }
readonly CKPTS_PATH="$GITHUB_WORKSPACE/llm_ckpts"
readonly CKPTS20_PATH="$GITHUB_WORKSPACE/llm_ckpts/20"
readonly CKPTS20_OUTPUT="${CKPTS20_PATH}/*.pt"
expected_num=21
exit_code=0

source ./ci_scripts/common/basic_func.sh

echo "start to test slurm training."

if [[ -d ${CKPTS20_PATH} ]]; then
    if ! rm -rf ${CKPTS20_PATH}/*; then
       echo "cleaning cached file in ${CKPTS20_PATH} failed, exit."
       exit 1
    fi
fi

srun -p llm --quotatype=spot -n 8 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ./ci_scripts/train/ci_7B_sft.py
[[ $? -ne 0 ]] && { echo "test slurm training failed.";  exit_code=$(($exit_code + 1)); }

num=$(num_files "${CKPTS20_OUTPUT}")
if [[ ${num} -ne ${expected_num} ]]; then
    echo "expect: ${expected_num} files, actual: ${num} files."
    exit_code=$(($exit_code + 1)) 
fi

# clean the test files.
if ! rm -rf ${CKPTS_PATH}/*; then
    echo "cleaning cached file in ${CKPTS_PATH} failed."
    exit_code=$(($exit_code + 1))
fi

exit $exit_code
