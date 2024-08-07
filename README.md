# Simulating DFAs
This repository contains code for my experiments on simulating deterministic finite automata (DFAs) using a bilinear model and various transformer architectures. The bilinear model and training script can be found in the `/src/bilinear` directory. The DFA class and corresponding OneHotEncoders can be found in the `/src/DFA` directory. The transformer models, including GT, WGT, and RWGT, as well as their training scripts can be found in the `/src/GT` directory.

## Training
I recommend referring to the training scripts for all training arguments. However, below are some example commands for training each type of model to simulate the ParityDFA.

To train a bilinear model, use a command such as: \
`python -m src.bilinear.train --DFA ParityDFA --num_epochs 20`

To train a GT model, use a command such as: \
`python -m src.GT.trainGT --DFA ParityDFA --num_epochs 20`

To train a WGT model, use a command such as: \
`python -m src.GT.trainWGT --DFA ParityDFA --num_epochs 20`

Finally, to train a RWGT model, use the WGT script but with the additional `rel` argument as follows: \
`python -m src.GT.trainWGT --rel True --DFA ParityDFA --num_epochs 20`