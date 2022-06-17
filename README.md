# Benefits of Additive Noise in Composing Classes with Bounded Capacity
by Alireza Fathollah Pour (fathola@mcmaster.ca) and Hassan Ashtiani (zokaeiam@mcmaster.ca).

This repository contains the codes used for training (noisy) networks and produce the empirical results of the paper, e.g., NVAC and accuracy plots.

Paper is available at https://arxiv.org/abs/2206.07199
## Guidlines
* Each folder is named according to the architecture; e.g., “250\_250\_250" is the folder containing results for a network with three hidden layers each containing 250 neurons.
* For each model there are three files included:
	1. results\_all.pckl: This file contains the results (e.g., errors, norms, etc.) for trained models with different noise standard deviations.
	2. valid\_results.pckl: The same as “results\_all.pckl” but only the models with highest validation accuracy are kept in this file.
	3. mixed\_covers.pckl: The results for NVAC of different approaches are stored in this file.
## Training
The training has been done on MNIST dataset. To train models from scratch execute the “train.py” model. To define the architecture pass the layers as a list to “layers” and use “noise\_level” to indicate what are the desired noise standard deviations. The “parent\_directory” is where the models are saved.
Note that if new models are trained in the same directory, the “valid\_results.pckl” and “all\_results.pckl” will be replaced with the new results.
## Evaluation
The “evaluate.py” contains the code to produce results in the paper, i.e., NVAC and error plots. To produce results for NVAC of a model, the “parent\_dir” should be set to the pickle file containing the covering number results (This pickle file will be generated as “mixed\_cover.pckl” if a new model is trained using the “train.py” file). Also the number of hidden layers (“depths”) and “widths” should be set accordingly.
> If the evaluation code is executed on a cpu only device the necessary workarounds is required to map the torch location to ‘cpu’ in order to open “mixed\_covers.pckl” files. Particularly,
	
	#CPU devices
	instead of pickle.load(f) use:
	CPU_Unpickler(f).load()
## Pre-trained Models
All the pre-trained models for the baseline architecture (i.e., three hidden layers each containing 250 neurons) are included in the “models” folder. For other architecures only the models trained with no noise and Gaussian noise with standard deviation of 0.05 are included. However, the pickle files containing the training results are included for every other architecture that is used to produce results.
## Results
Please refer to Figures 1 and 2 (Appendix I) in the paper.
