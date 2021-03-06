#!/bin/bash



cell_model=$1
config=$2
seed=$3
tstop=$4
feinp_min=$5
feinp_max=$6
fiinp_0=$7 # lower and upper bound around pref * exc freq
pref=$8 # prefactor for the inhibitory freq
incr=$9

### Here we sample on the excitatory and around two times the same freq for inhibitory

start_e=$feinp_min*10 # because increment cannot be float and thus everything is multiplied by 2
final_e=$feinp_max*10 

count=0


for ((e=$start_e;e<=$final_e;e+=1))
  do 
	feinp=$(bc <<< "scale=4;$e/$incr")
	echo -e "\n fe = $feinp \n"
	
	#test_a1=$(bc <<< "scale=0;$e/$incr*$pref*10-$fiinp*10") #underscore a because test alone was not accepted 
	
	#test_a=${test_a1/.*} ## transforms to int  
	
	echo -e "\n test = $test_a \n"

	#if [ $test_a -lt 0 ]; then 
		
	#	start_i=0
	#else
	#	start_i0=$(bc <<< "scale=4;$feinp*$pref*10-$fiinp*10")
	#	start_i=${start_i0/.*}
	#	echo -e "\n starti0 = $start_i0   start_i = $start_i"

	#fi
		
	final_i0=$(bc <<< "scale=4;$feinp*$pref*10+$fiinp_0*10")
	final_i=${final_i0/.*}

	start_i0=$(bc <<< "scale=4;$feinp*$pref*10")
	start_i=${start_i0/.*}

	echo -e "start_i = $start_i \n"
	echo -e "final_i = $final_i \n"	


	for ((i=$start_i;i<=$final_i;i+=1))
	  do
		
		fiinp=$(bc <<< "scale=4;$i/$incr")
		echo -e "fi = $fiinp \n"
		python New_tf.py $cell_model $config -s --SEED $seed --tstop $tstop --fe_inp $feinp --fi_inp $fiinp
		
		sleep 2 # defines implicitly the # of codes that run at the same time 60 >~ 2 codes
	  	
		count=$(bc <<< "scale=4;$count+1")

		if [ $count -gt 63 ]; then

			echo -e "break"
			sleep 120
			count=0
			echo -e "\n INSIDE IF LOOP \n"

		fi	
		
	
	done
 done



#
