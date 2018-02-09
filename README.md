# MiniWord2Vec

This application is an implementation of both the skipgram and cbow techniques used in the Word2Vec algorithm.
 

# Requirements

Support for Python 2 and 3. Install the package requirements via
```
pip install -r requirements.txt
```
## Note
Requires cupy to run on GPU for fast computations, and that is the default behaviour.
 ```cupy``` requires CUDA related libraries, cuDNN and NCCL, to be installed before installing CuPy.

Replace ```import cupy as np``` with ```import numpy as np``` if you wish to run it on the CPU.  
 
# Data
 
The training data can be found in the ```data/``` folder.
 
 
# Usage

For training, use the run script.
For CBoW, use:
```
./run cbow
```

For Skipgram, use:
```
./run skipgram
```

# Tuning Parameters
You can edit the parameters by specifying their values in the ```run``` file.
Parameters that can be edited:
- Dimension of the word embedding, default: 300
- No. of epochs to train the data on, default: 100
- Window size for CBoW, default: 3

~                             