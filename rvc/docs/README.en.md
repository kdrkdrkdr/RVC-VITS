<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
An easy-to-use Voice Conversion framework based on VITS.<br><br>

[![madewithlove](https://forthebadge.com/images/badges/built-with-love.svg)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)
  
<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>
  
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/github/license/liujing04/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/%E4%BD%BF%E7%94%A8%E9%9C%80%E9%81%B5%E5%AE%88%E7%9A%84%E5%8D%8F%E8%AE%AE-LICENSE.txt)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

</div>

------
[**Changelog**](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/Changelog_CN.md) | [**FAQ (Frequently Asked Questions)**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/FAQ-(Frequently-Asked-Questions)) 

[**English**](./README.en.md) | [**中文简体**](../README.md) | [**日本語**](./README.ja.md) | [**한국어**](./README.ko.md) ([**韓國語**](./README.ko.han.md))

> Check our [Demo Video](https://www.bilibili.com/video/BV1pm4y1z7Gm/) here!

> Realtime Voice Conversion Software using RVC : [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

> The dataset for the pre-training model uses nearly 50 hours of high quality VCTK open source dataset.

> High quality licensed song datasets will be added to training-set one after another for your use, without worrying about copyright infringement.
## Summary
This repository has the following features:
+ Reduce tone leakage by replacing source feature to training-set feature using top1 retrieval;
+ Easy and fast training, even on relatively poor graphics cards;
+ Training with a small amount of data also obtains relatively good results (>=10min low noise speech recommended);
+ Supporting model fusion to change timbres (using ckpt processing tab->ckpt merge);
+ Easy-to-use Webui interface;
+ Use the UVR5 model to quickly separate vocals and instruments.
## Preparing the environment
We recommend you install the dependencies through poetry.

The following commands need to be executed in the environment of Python version 3.8 or higher:
```bash
# Install PyTorch-related core dependencies, skip if installed
# Reference: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

#For Windows + Nvidia Ampere Architecture(RTX30xx), you need to specify the cuda version corresponding to pytorch according to the experience of https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/issues/21
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install the Poetry dependency management tool, skip if installed
# Reference: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# Install the project dependencies
poetry install
```
You can also use pip to install the dependencies

```bash
pip install -r requirements.txt
```

## Preparation of other Pre-models
RVC requires other pre-models to infer and train.

You need to download them from our [Huggingface space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/).

Here's a list of Pre-models and other files that RVC needs:
```bash
hubert_base.pt

./pretrained 

./uvr5_weights

If you want to test the v2 version model (the v2 version model has changed the input from the 256 dimensional feature of 9-layer Hubert+final_proj to the 768 dimensional feature of 12-layer Hubert, and has added 3 period discriminators), you will need to download additional features

./pretrained_v2

#If you are using Windows, you may also need this dictionary, skip if FFmpeg is installed
ffmpeg.exe
```
Then use this command to start Webui:
```bash
python infer-web.py
```
If you are using Windows, you can download and extract `RVC-beta.7z` to use RVC directly and use `go-web.bat` to start Webui.

There's also a tutorial on RVC in Chinese and you can check it out if needed.

## Credits
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
## Thanks to all contributors for their efforts

<a href="https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=liujing04/Retrieval-based-Voice-Conversion-WebUI" />
</a>

