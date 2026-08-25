#!/bin/bash
run_flag=$1



request_memory="40GB"
CPUS="1"
runs="1"

wd=`pwd`

#architectures="simple_model simple_model_ReLu step_model step_model_ReLu UNet_model_ResNet ResNet_model ResNet_model_ReLu ResNet_model_big ResStep_model_Tanh ResStepDown_model_Tanh"
sample_sizes="5000"


#architectures="ResStepDown_big_500 ResStepDown_big_1200 ResStepDown_big_2400 ResStepDown_big_3600 ResStepDown_layers_500 ResStepDown_layers_1000 ResStepDown_layers_5000 ResStepDown_simple_500 ResStepDown_simple_1000 ResStepDown_simple_5000 ResStepDown_simple+_7000 ResStepDown_val_500 ResStepDown_val_1000 ResStepDown_val_5000 ResStepDown_simple+_7000"
architectures="RL_UNet_cat_ReLu RL_UNet_model_ResNet_ReLu ResStepDown_train"


in_fns="1"

for run in ${runs}; do
for architecture in ${architectures}; do
for sample_size in ${sample_sizes}; do
    wdir="/tikhome/lspatscheck/Documents/bsc/simulation_data/CNN_training/Results/model_${architecture}/sample_size_${sample_size}/futher"
    mkdir -p $wdir
    cd $wdir
    wdir_path=`pwd`
    echo $wdir_path
    con_file="JOB.condor"

    OUTPUT_FILE=${wdir_path}/outfile.pkl
	
    echo "universe = vanilla" > $con_file
    echo "request_CPUs = ${CPUS}" >> $con_file
    echo "request_memory = ${request_memory}" >> $con_file
    echo "executable = /usr/bin/mpiexec" >> $con_file
    echo "arguments = -n ${CPUS} python3 /tikhome/lspatscheck/Documents/bsc/simulation_scripts/${architecture}.py --sample_size ${sample_size}" >> $con_file
    echo "output = ${wdir_path}/condor.out" >> $con_file
    echo "error = ${wdir_path}/condor.err" >> $con_file
    echo "log = ${wdir_path}/condor.log" >> $con_file
    echo "getenv = true" >> $con_file
    echo "queue" >> $con_file
    if [ -f "$OUTPUT_FILE" ]; then
	    echo "$OUTPUT_FILE exists."
    else
	    if [ "${run_flag}" == "condor" ]; then
		    	condor_submit $con_file -batch-name big_training
	    fi
    fi
    cd ..

    cd $wd

done
done
done
exit 0