cd rvc
mkdir "../vits_dataset/$1"
python inference.py $1 $2
cd ..