#!/bin/bash
# Prepare (and optionally submit) HTCondor jobs for training the CNNs.
#
#   bash condor/condor_training.sh           # only write the job files
#   bash condor/condor_training.sh condor    # write and submit them
#
# One job per (model, sample size), running scripts/train.py. Results:
# $OUT_DIR/<model>/sample_size_<n>/. Jobs whose DONE file exists are skipped.
# Requires the package to be installed (bash condor/build.sh).
#
# Settings (environment variables):
#   DATA_DIR        directory containing train_data/   (required)
#   OUT_DIR         output directory                   (default: ./training_results)
#   MODELS          shallow, deep and/or unet          (default: "unet")
#   SAMPLE_SIZES    training set sizes                 (default: "5000")
#   REQUEST_MEMORY  memory per job                     (default: 40GB)
#   CPUS            CPUs per job                       (default: 1)
run_flag=$1

if [ -z "${DATA_DIR}" ]; then
    echo "Please set DATA_DIR to the directory containing train_data/." >&2
    exit 1
fi
OUT_DIR=${OUT_DIR:-$PWD/training_results}
models=${MODELS:-"unet"}
sample_sizes=${SAMPLE_SIZES:-"5000"}
request_memory=${REQUEST_MEMORY:-"40GB"}
CPUS=${CPUS:-1}
repo_dir=$(cd "$(dirname "$0")/.." && pwd)

for model in ${models}; do
for sample_size in ${sample_sizes}; do
    wdir="${OUT_DIR}/${model}/sample_size_${sample_size}"
    mkdir -p "$wdir"
    wdir_path=$(cd "$wdir" && pwd)
    echo "$wdir_path"
    con_file="${wdir_path}/JOB.condor"

    {
        echo "universe = vanilla"
        echo "request_CPUs = ${CPUS}"
        echo "request_memory = ${request_memory}"
        echo "executable = /usr/bin/mpiexec"
        echo "arguments = -n ${CPUS} python3 ${repo_dir}/scripts/train.py --model ${model} --sample-size ${sample_size} --data-dir ${DATA_DIR} --output-dir ${wdir_path}"
        echo "initialdir = ${wdir_path}"
        echo "output = ${wdir_path}/condor.out"
        echo "error = ${wdir_path}/condor.err"
        echo "log = ${wdir_path}/condor.log"
        echo "getenv = true"
        echo "queue"
    } > "$con_file"

    # scripts/train.py writes DONE when all runs have finished
    if [ -f "${wdir_path}/DONE" ]; then
        echo "${wdir_path}/DONE exists."
    elif [ "${run_flag}" == "condor" ]; then
        condor_submit "$con_file" -batch-name training
    fi
done
done
exit 0
