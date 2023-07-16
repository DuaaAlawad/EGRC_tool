#   EGRC: Effective inferring Gene Regulatory network using graph convolution network with SAGpooling layer.


### Table of Content

- [Setup](#getting-started)
- [Dataset](#Dataset)
- [Download and install code](#download-and-install-code)
- [Demo](#demo)

  
# Getting Started.. 
 
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

 ## Dataset
The dataset can be found in the dataset directory. 
the data which have been used in this paper is from DREAM5 challenges.

In **DREAM5 challenge**, the dataset  comprises three main files:

    1-  Gene expression data
    2-  GoldStandardard for each network
    3-  List of  transcription factors (TFs)


|    Network    |    Organism      | # TranscFactors  | # Genes | # Chips |
| ------------- | ---------------- | -----------------| --------|---------|
|    Network 1  |   In silico      |        195       |   1643  |   805   |
|    Network 3  |   E. coli        |        334       |   4511  |   805   |
|    Network 4  |   S. cerevisiae  |        333       |   5950  |   536   |



  
## Download and install code

- Retrieve the code

```
git clone https://github.com/DuaaAlawad/EGRC_tool.git
```

## Demo

To run the program, first, set the input path in the input.txt file. Here is a sample input file from DREAM5 E-coli Dataset.

```
/home/dmalawad/Research/EGRC
data3
```


Then, run following python command from the root directory.

### Option 1 : (Recommended) Use python virutal environment with conda（<https://anaconda.org/>）

```shell
conda create -n EGRCEnv python=3.7.9 pip
conda activate EGRCEnv
conda install -c anaconda tensorflow-gpu
pip install -r requirements.txt
```



### Option 2 :  Use poetry to setup the virutal environment with conda
You would need to install the following software before replicating this framework in your local or server machine.

 ```
- Python version 3.7.9
- Poetry version 1.1.12
  You can install poetry by running the following command:
     curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
To configure your current shell run `source $HOME/.poetry/env`
```

``` 
cd EGRC
poetry install
poetry run python EGRC.py
```

Finally, check **output** folder for results. The output directory contains the prob folder, which contains the importance scores in CSV files. In addition, there is a folder holding the same name as input data that contains  The OutputResults.txt file showing the results in AUROC and AUPR.


## Authors

Duaa Mohammad Alawad, Ataur Katebi, Md Tamjidul Hoque* . For any issue please contact: Md Tamjidul Hoque, thoque@uno.edu 
