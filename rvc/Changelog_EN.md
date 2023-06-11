### 2023-05-28
- Add v2 jupyter notebook, korean changelog, fix some environment requirments
- Add voiceless consonant and breath protection mode
- Support crepe-full pitch detect
- UVR5 vocal separation: support dereverb models and de-echo models
- Add experiment name and version on the name of index
- Support users to manually select export format of output audios when batch voice conversion processing and UVR5 vocal separation
- 32k model training is no more supported

todolist：
- Feature retrieval: add temporal feature retrieval
- Feature retrieval: add pre-kmeans option
- Feature retrieval: add PCAR dimensionality reduction
- Add onnx inference support
- Random data augmentation when training: pitch, gender, eq, noise
- Add v2 version pretrained-models

### 2023-05-13
- Clear the redundant codes in the old version of runtime in the one-click-package: infer_pack and uvr5_pack
- Fix pseudo multiprocessing bug in training set preprocessing
- Adding median filtering radius adjustment for harvest pitch recognize algorithm
- Support post processing resampling for exporting audio
- Multi processing "n_cpu" setting for training is changed from "f0 extraction" to "data preprocessing and f0 extraction"
- Automatically detect the index paths under the logs folder and provide a drop-down list function
- Add "Frequently Asked Questions and Answers" on the tab page (you can also refer to github RVC wiki)
- When inference, harvest pitch is cached when using same input audio path (purpose: using harvest pitch extraction, the entire pipeline will go through a long and repetitive pitch extraction process. If caching is not used, users who experiment with different timbre, index, and pitch median filtering radius settings will experience a very painful waiting process after the first inference)

### 2023-05-14
- Use volume envelope of input to mix or replace the volume envelope of output (can alleviate the problem of "input muting and output small amplitude noise". If the input audio background noise is high, it is not recommended to turn it on, and it is not turned on by default (1 can be considered as not turned on)
- Support saving extracted small models at a specified frequency (if you want to see the performance under different epochs, but do not want to save all large checkpoints and manually extract small models by ckpt-processing every time, this feature will be very practical)
- Resolve the issue of "connection errors" caused by the server's global proxy by setting environment variables
- Supports pre-trained v2 models (currently only 40k versions are publicly available for testing, and the other two sampling rates have not been fully trained yet)
- Limit excessive volume exceeding 1 before inference
- Slightly adjusted the settings of training-set preprocessing


#######################

History changelogs:

### 2023-04-09
- Fixed training parameters to improve GPU utilization rate: A100 increased from 25% to around 90%, V100: 50% to around 90%, 2060S: 60% to around 85%, P40: 25% to around 95%; significantly improved training speed
- Changed parameter: total batch_size is now per GPU batch_size
- Changed total_epoch: maximum limit increased from 100 to 1000; default increased from 10 to 20
- Fixed issue of ckpt extraction recognizing pitch incorrectly, causing abnormal inference
- Fixed issue of distributed training saving ckpt for each rank
- Applied nan feature filtering for feature extraction
- Fixed issue with silent input/output producing random consonants or noise (old models need to retrain with a new dataset)

### 2023-04-16 Update
- Added local real-time voice changing mini-GUI, start by double-clicking go-realtime-gui.bat
- Applied filtering for frequency bands below 50Hz during training and inference
- Lowered the minimum pitch extraction of pyworld from the default 80 to 50 for training and inference, allowing male low-pitched voices between 50-80Hz not to be muted
- WebUI supports changing languages according to system locale (currently supporting en_US, ja_JP, zh_CN, zh_HK, zh_SG, zh_TW; defaults to en_US if not supported)
- Fixed recognition of some GPUs (e.g., V100-16G recognition failure, P4 recognition failure)

### 2023-04-28 Update
- Upgraded faiss index settings for faster speed and higher quality
- Removed dependency on total_npy; future model sharing will not require total_npy input
- Unlocked restrictions for the 16-series GPUs, providing 4GB inference settings for 4GB VRAM GPUs
- Fixed bug in UVR5 vocal accompaniment separation for certain audio formats
- Real-time voice changing mini-GUI now supports non-40k and non-lazy pitch models

### Future Plans:
Features:
- Add option: extract small models for each epoch save
- Add option: export additional mp3 to the specified path during inference
- Support multi-person training tab (up to 4 people)

Base model:
- Collect breathing wav files to add to the training dataset to fix the issue of distorted breath sounds
- We are currently training a base model with an extended singing dataset, which will be released in the future
