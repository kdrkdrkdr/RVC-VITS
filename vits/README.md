# MB-iSTFT-VITS with Multilingual Implementations
<img src="./fig/with_tsukuyomi_chan.png" width="100%">

This is an multilingual implementation of [MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS) to support conversion to various languages. MB-iSTFT-VITS showed 4.1 times faster inference time compared with original VITS! </br>
Preprocessed Japanese Single Speaker training material is provided with [つくよみちゃんコーパス(tsukuyomi-chan corpus).](https://tyc.rei-yumesaki.net/material/corpus/) You need to download the corpus and place 100 `.wav` files to `./tsukuyomi_raw`. 
</br>

- Currently Supported: Japanese / Korean
- Chinese / CJKE / and other languages will be updated very soon!


# How to use
Python >= 3.6 (Python == 3.7 is suggested)

## Clone this repository
```sh
git clone https://github.com/misakiudon/MB-iSTFT-VITS-multilingual.git
```

## Install requirements
```sh
pip install -r requirements.txt
```
You may need to install espeak first: `apt-get install espeak`

## Create manifest data
### Single speaker
"n_speakers" should be 0 in config.json
```
path/to/XXX.wav|transcript
```
- Example
```
dataset/001.wav|こんにちは。
```

### Mutiple speakers
Speaker id should start from 0 
```
path/to/XXX.wav|speaker id|transcript
```
- Example
```
dataset/001.wav|0|こんにちは。
```

## Preprocess
Japanese preprocessed manifest data is provided with `filelists/filelist_train2.txt.cleaned` and `filelists/filelist_val2.txt.cleaned`.
```sh
# Single speaker
python preprocess.py --text_index 1 --filelists path/to/filelist_train.txt path/to/filelist_val.txt --text_cleaners 'japanese_cleaners'

# Mutiple speakers
python preprocess.py --text_index 2 --filelists path/to/filelist_train.txt path/to/filelist_val.txt --text_cleaners 'japanese_cleaners'
```

If your speech file is either not `22050Hz / Mono / PCM-16`, the you should resample your .wav file first. 
```sh
python convert_to_22050.py --in_path path/to/original_wav_dir/ --out_path path/to/output_wav_dir/
```

## Build monotonic alignment search
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```

## Setting json file in [configs](configs)

| Model | How to set up json file in [configs](configs) | Sample of json file configuration|
| :---: | :---: | :---: |
| iSTFT-VITS | ```"istft_vits": true, ```<br>``` "upsample_rates": [8,8], ``` | ljs_istft_vits.json |
| MB-iSTFT-VITS | ```"subbands": 4,```<br>```"mb_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ``` | ljs_mb_istft_vits.json |
| MS-iSTFT-VITS | ```"subbands": 4,```<br>```"ms_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ``` | ljs_ms_istft_vits.json |

For tutorial, check `config/tsukuyomi_chan.json` for more examples
- If you have done preprocessing, set "cleaned_text" to true. 
- Change `training_files` and `validation_files` to the path of preprocessed manifest files. 
- Select same `text_cleaners` you used in preprocessing step. 

## Train
```sh
# Single speaker
python train_latest.py -c <config> -m <folder>

# Mutiple speakers
python train_latest_ms.py -c <config> -m <folder>
```
In the case of training MB-iSTFT-VITS with Japanese tutorial corpus, run the following script. Resume training from lastest checkpoint is automatic.
```sh
python train_latest.py -c configs/tsukuyomi_chan.json -m tsukuyomi
```

After the training, you can check inference audio using [inference.ipynb](inference.ipynb)

## References
- https://github.com/MasayaKawamura/MB-iSTFT-VITS
- https://github.com/CjangCjengh/vits
- https://github.com/Francis-Komizu/VITS
