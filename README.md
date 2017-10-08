# Node2vec with tensorflow
This repo contains ad hoc implementation of metapath2vec using tensorflow. I call it ad hoc because the codes are not so clean and efficient. However, it is applicable for large networks. I tested on a network with ----- nodes. 
  
main reference appeared at KDD 2016: [metapath2vec: ](http://arxiv.org/abs/xyz)
  
Also, I noticed that the first author of the paper open sourced the implementation. I guess that is more efficent. So please try to use that first. This repo is for people who want to use/study tensorflow for some reasons. 
  
## Requirements
I recommend you to install [Anaconda](https://www.continuum.io/downloads) and then tensorflow.
- [tneosorflow](http://tensorflow.org)
- and some other libraries...

## How to train.
learn embeddings using the random walks
```
python main.py --walks ./data/test_data/random_walks.txt --types ./data/test_data/node_type_mapings.txt --log ./log --negative-samples 5 --window 1 --epochs 100 --care-type 0
python main.py --walks ./data/test_data/random_walks.txt --types ./data/test_data/node_type_mapings.txt --log ./log --negative-samples 1 --window 1 --epochs 100 --care-type 1
tensorboard --logdir=./log/
```

## how to load the learned embeddings 
```
import numpy as np
import json
index2nodeid = json.load(open("./log/index2nodeid.json"))
index2nodeid = {int(k):v for k,v in index2nodeid.items()}
nodeid2index = {v:int(k) for k,v in index2nodeid.items()}
node_embeddings = np.load("./log/node_embeddings.npz")['arr_0']

#node embeddings of "yi"
node_embeddings[nodeid2index["yi"]]
```

##To do list

