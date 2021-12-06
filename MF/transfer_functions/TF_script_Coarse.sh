#!/bin/bash



cell_model=$1
config=$2
seed=$3
tstop=$4
feinp_min=$5
feinp_max=$6
fiinp_min=$7
fiinp_max=$8
incr=$9

start_e=$feinp_min*2 # because increment cannot be float and thus everything is multiplied by 2
final_e=$feinp_max*2 

start_i=$fiinp_min*2
final_i=$fiinp_max*2


for ((e=$start_e;e<=$final_e;e+=1))
  do 
	feinp=$(bc <<< "scale=4;$e/$incr")
	echo -e "\n fe = $feinp \n"
	
	for ((i=$start_i;i<=$final_i;i+=1))
	  do
		
		fiinp=$(bc <<< "scale=4;$i/$incr")
		echo -e "fi = $fiinp \n"
		python New_tf.py $cell_model $config -s --SEED $seed --tstop $tstop --fe_inp $feinp --fi_inp $fiinp
		
		sleep 40 # defines implicitly the # of codes that run at the same time 60 >~ 2 codes
	  done
 done



#
