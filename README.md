# Darin

Necessary files:

1. finalNN.txt (my network)
2. GAME.py (exeutable file)

To execute: with all installed packages run 
`$ python3 test.py`

with GAME.py written as one of the agents, where test.py and all others can be found here:
https://github.com/dasimagin/renju (to run just download all this repo as .zip file, than extract to folder which contains GAME.py and finalNN.txt)


Other files:

1. renju.ipynb (all my work with Darin in ipynb format)
2. /labs/???.ipynb (all labs, given by our mentor)
3. /samples/???.? (all required samples)

Headers for Darin:
1. Pytorch 1.0.1 
`$ conda install pytorch torchvision cudatoolkit=9.0 -c pytorch`
2. Python 3.5 and newer
Already in core of Ubuntu 16.04
3. Numpy (any)
`$ sudo pip3 install numpy`
4. sys
Basic library

Learning code:
Directly in renju.ipynb, dataWhole.txt is dataset from here: https://github.com/dasimagin/renju/blob/master/data/train-1.tar.xz
but excluding all draw and unknown status games.

Extracting all of this is here
