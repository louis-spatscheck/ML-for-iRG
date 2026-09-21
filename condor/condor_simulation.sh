#!/bin/bash
# Prepare (and optionally submit) HTCondor jobs for the Ising simulations.
#
#   bash condor/condor_simulation.sh           # only write the job files
#   bash condor/condor_simulation.sh condor    # write and submit them
#
# One job per (lattice size, betaJ). Results: $OUT_DIR/lattice_size<L>/betaJ<b>/data.gz
# Jobs whose data.gz already exists are skipped.
# Requires the package to be installed (bash condor/build.sh).
#
# Settings (environment variables):
#   OUT_DIR         output directory       (default: ./simulation_data)
#   LATTICE_SIZES   lattice sizes          (default: "64 128 256")
#   BETAJS          inverse temperatures   (default: "0.3 0.5")
#   REQUEST_MEMORY  memory per job         (default: 100GB)
#   CPUS            CPUs per job           (default: 1)
run_flag=$1

OUT_DIR=${OUT_DIR:-$PWD/simulation_data}
lattice_sizes=${LATTICE_SIZES:-"64 128 256"}
betaJs=${BETAJS:-"0.3 0.5"}
request_memory=${REQUEST_MEMORY:-"100GB"}
CPUS=${CPUS:-1}

for betaJ in ${betaJs}; do
for lattice_size in ${lattice_sizes}; do
    wdir="${OUT_DIR}/lattice_size${lattice_size}/betaJ${betaJ}"
    mkdir -p "$wdir"
    wdir_path=$(cd "$wdir" && pwd)
    echo "$wdir_path"
    con_file="${wdir_path}/JOB.condor"

    {
        echo "universe = vanilla"
        echo "request_CPUs = ${CPUS}"
        echo "request_memory = ${request_memory}"
        echo "executable = /usr/bin/mpiexec"
        echo "arguments = -n ${CPUS} python3 -m invrg.simulation --lattice_size ${lattice_size} --betaJ ${betaJ}"
        echo "initialdir = ${wdir_path}"
        echo "output = ${wdir_path}/condor.out"
        echo "error = ${wdir_path}/condor.err"
        echo "log = ${wdir_path}/condor.log"
        echo "getenv = true"
        echo "queue"
    } > "$con_file"

    # the simulation writes data.gz into its initial directory when it has finished
    if [ -f "${wdir_path}/data.gz" ]; then
        echo "${wdir_path}/data.gz exists."
    elif [ "${run_flag}" == "condor" ]; then
        condor_submit "$con_file" -batch-name lattice_test
    fi
done
done
exit 0
