# Mo-PAE

Here is the code for the implementation of Mo-PAE network.
Due to some data security, we only share a sample data of Geolife (preprocessed already), in the sample_trainingdata folder.

## Training Setting for the results in the paper:
The main goal of the proposed adversarial network is to learn an efficient feature representation based on the utility and privacy budgets, using all users' mobility histories. In most experiments in this work, the trajectory sequences consist of 10 historical locations with timestamps, and the impact of the varying sequence lengths is discussed.
After data pre-processing, 80\% of each user's records are segmented as the training set and the remaining 20\% as the testing set. We utilize the mini-batch learning method with the size of 128 to train the model until the expected convergence. We take a gradient step to optimize the sum loss in terms of L_R, L_U, and L_P concurrently. Meanwhile, the L_sum is optimized by using the Adam optimizer. 

All the experiments are performed with the Tesla V100 GPU; 
a round of training would take 30 seconds on average; 
and each experiment trains for 1000 rounds.
