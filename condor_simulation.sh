#!/bin/bash
run_flag=$1


lattice_sizes="64 128 256"
betaJs="0.3 0.5"

request_memory="100GB"
CPUS="1"
runs="1"

wd=`pwd`

in_fn="/tikhome/lspatscheck/Documents/bsc/simulation_scripts/simulation.py"

for run in ${runs}; do
for betaJ in ${betaJs}; do
for lattice_size in ${lattice_sizes}; do
    wdir="/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size${lattice_size}/betaJ${betaJ}"
    mkdir -p $wdir
    cd $wdir
    wdir_path=`pwd`
    echo $wdir_path
    con_file="JOB.condor"

    OUTPUT_FILE=${wdir_path}/what.pkl
	
    echo "universe = vanilla" > $con_file
    echo "request_CPUs = ${CPUS}" >> $con_file
    echo "request_memory = ${request_memory}" >> $con_file
    echo "executable = /usr/bin/mpiexec" >> $con_file
    echo "arguments = -n ${CPUS} python3 ${in_fn} --lattice_size ${lattice_size} --betaJ ${betaJ}" >> $con_file
    echo "output = ${wdir_path}/condor.out" >> $con_file
    echo "error = ${wdir_path}/condor.err" >> $con_file
    echo "log = ${wdir_path}/condor.log" >> $con_file
    echo "getenv = true" >> $con_file
    echo "queue" >> $con_file
    if [ -f "$OUTPUT_FILE" ]; then
	    echo "$OUTPUT_FILE exists."
    else
	    if [ "${run_flag}" == "condor" ]; then
		    	condor_submit $con_file -batch-name lattice_test
	    fi
    fi
    cd ..

    cd $wd

done
done
done
exit 0