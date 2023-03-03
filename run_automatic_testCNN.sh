#!/bin/bash

tf= ( 0 0 )
load_model= ( 0 0 )
save_model= ( 1 1 )

for bayes in 0 1
do
    #bayes = 1
    if [ $bayes -eq 1 ];
    then
         echo "**************************************************************************"
         echo "**********************  Starting Bayes Approach  *************************"
         echo "**************************************************************************"
    fi
    sudo python -u /home/rdasilv2/CNN_paper/Scripts/CNN_Simulated_Data.py $bayes ${tf[$bayes]} ${load_model[$bayes]} ${save_model[$bayes]} & PIDY=$!
    wait $PIDY
done
exit 0
