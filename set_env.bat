pip install -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
cd vits/monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
cd ../../