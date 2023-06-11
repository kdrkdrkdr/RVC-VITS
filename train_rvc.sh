cd rvc
mkdir "./logs/$1"
python trainset_preprocess_pipeline_print.py ../rvc_dataset/$1 40000 16 ./logs/$1 False
python extract_f0_print.py ./logs/$1 16 harvest
python extract_feature_print.py cuda:0 1 0 0 ./logs/$1 v2
python write_filelist.py $1
python train_nsf_sim_cache_sid_load_pretrain.py -e %1 -sr 40k -f0 1 -bs 12 -g 0 -te $2 -se $2 -pg f0G40k-latest.pth -pd f0D40k-latest.pth -l 0 -c 0 -sw 0 -v v2
cd ..