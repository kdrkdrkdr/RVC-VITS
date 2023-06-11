# Few-shot multilingual-tts with rvc and vits
A tool that allows you to learn soft multilingual speech with a small amount of data set (5-10 minutes) using RVC.
Most speech synthesis models require vast amounts of data.
However, it is not always possible to learn only in situations where there is a lot of data.
This repository started with the idea of "Then why don't we clone a dataset and use it?"

### 0.Process
0. RVC Training with few dataset
0. Dataset Cloning with Trained RVC Model.
0. Training Vits
0. Inference

### 1. Pre-requisites
0. Python >= 3.8
0. Download this repository's release
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Build requirements.txt and torch
```sh
./set_env.sh
```
0. Put the dataset into the rvc_dataset directory according to the following file structure. In this experiment, I used 50 wavs file of ljspeech datasets (330seconds).
```sh
rvc_dataset
├───ljs
│   ├───LJ001-0001.wav
│   ├───LJ001-0002.wav
│   ├───...
│   └───LJ001-0050.wav
```


### 2. Training
```sh
./train_rvc.sh ljs 500
# If you want to train korean tts, change ja to ko (ja -> japanese, ko -> korean, en -> english)
./make_dataset.sh ljs ja
./train_vtis.sh ljs 
```


### 3. Inference
See vits/inference.ipynb


## Test Datasets
| Language | Name | Link |
| --- | --- | --- |
| Japanese | JSUT | https://sites.google.com/site/shinnosuketakamichi/publication/jsut | - |
| English | LJSPEECH | https://keithito.com/LJ-Speech-Dataset/ | - |
| Korean | KSS | https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset | - |



## References
- https://github.com/jaywalnut310/vits.git
- https://github.com/MasayaKawamura/MB-iSTFT-VITS.git
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
- https://arxiv.org/pdf/2206.12132.pdf
