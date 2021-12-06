#!/bin/bash



cell_model=$1
config=$2
seed=$3
tstop=$4
feinp=$5
fiinp=$6


python New_tf_multiprocess.py $cell_model $config -s --SEED $seed --tstop $tstop --fe_inp $feinp --fi_inp $fiinp




#
