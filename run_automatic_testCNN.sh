#!/bin/bash

# transfer learning, 0 - False, 1 - True
tf= ( 0 0 )
# load previous stored model weights, 0 - False, 1 - True
load_model= ( 0 0 )
save_model= ( 1 1 )
your_path= '<here>'

for bayes in 0 1
do
    #bayes = 1
    if [ $bayes -eq 1 ];
    then
         echo "**************************************************************************"
         echo "**********************  Starting Bayes Approach  *************************"
         echo "**************************************************************************"
    fi
    sudo python -u ${your_path}/Scripts/CNN_Simulated_Data.py $bayes ${tf[$bayes]} ${load_model[$bayes]} ${save_model[$bayes]} & PIDY=$!
    wait $PIDY
done
exit 0
