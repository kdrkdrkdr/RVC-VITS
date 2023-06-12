import torch, os, traceback, sys, warnings, shutil, numpy as np
from infer_pack.models import SynthesizerTrnMs768NSFsid
import soundfile as sf
from fairseq import checkpoint_utils
from vc_infer_pipeline import VC
from config import Config
from my_utils import load_audio

model_name = 'ljs'
weight_root = "weights"
cpt, tgt_sr, if_f0, version, net_g, vc, n_spk = None, None, None, None, None, None, None

config = Config()
hubert_model = None
now_dir = os.getcwd()
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()



def load_rvc_model(model_name):
    global cpt, tgt_sr, if_f0, version, net_g, vc, n_spk
    cpt = torch.load(f'weights/{model_name}.pth', map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0] 
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)

    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]

def vc_single(
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
):
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            f0_file=f0_file,
        )
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def vc_multi(
    sid,
    dir_path,
    opt_root,
    paths,
    f0_up_key,
    f0_method,
    file_index,
    file_index2,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    format1
):
    dir_path = (
        dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    )
    opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    os.makedirs(opt_root, exist_ok=True)
    try:
        if dir_path != "":
            paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
        else:
            paths = [path.name for path in paths]
    except:
        traceback.print_exc()
        paths = [path.name for path in paths]

    for path in paths:
        _, opt = vc_single(
            sid,
            path,
            f0_up_key,
            None,
            f0_method,
            file_index,
            file_index2,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect,
        )
        tgt_sr, audio_opt = opt
        
        sf.write(
            "%s/%s" % (opt_root, os.path.basename(path)),
            audio_opt,
            tgt_sr,
        )


symbols_format = """
{}

symbols = [_pad] + list(_punctuation) + list(_letters)

SPACE_ID = symbols.index(" ")
"""

symbols_ko = '''
_pad        = '_'
_punctuation = ',.!?…~'
_letters = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅓㅗㅜㅡㅣㅐㅔ '
'''
symbols_ja = """
_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'AEINOQUabdefghijkmnoprstuvwyzʃʧʦ↓↑ '
"""
symbols_en = """
_pad        = '_'
_punctuation = ';:,.!?¡¿—…\"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ'
"""


vits_config_format = '''{{
  "train": {{
    "log_interval": 200,
    "eval_interval": 1000,
    "seed": 1234,
    "epochs": 20000,
    "learning_rate": 2e-4,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "batch_size": 32,
    "fp16_run": false,
    "lr_decay": 0.999875,
    "segment_size": 8192,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0,
    "fft_sizes": [384, 683, 171],
    "hop_sizes": [30, 60, 10],
    "win_lengths": [150, 300, 60],
    "window": "hann_window"  
  }},
  "data": {{
    "training_files":"../vits_dataset/{}/train.txt.cleaned",
    "validation_files":"../vits_dataset/{}/val.txt.cleaned",
    "text_cleaners":["{}"],
    "max_wav_value": 32768.0,
    "sampling_rate": 40000,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 80,
    "mel_fmin": 0.0,
    "mel_fmax": null,
    "add_blank": true,
    "n_speakers": 0,
    "cleaned_text": true
  }},
  "model": {{
    "ms_istft_vits": true,
    "mb_istft_vits": false,
    "istft_vits": false,
    "subbands": 4,
    "gen_istft_n_fft": 16,
    "gen_istft_hop_size": 4,
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "upsample_rates": [4,4],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16,16],
    "n_layers_q": 3,
    "use_spectral_norm": false,
    "use_sdp": false
  }}
}}'''


def make_dataset(vc_model_name='ljs', language='ja'):
    load_hubert()
    load_rvc_model(vc_model_name)
    # return
    if language == 'ko':
        dataset_name = 'kss'
        cleaners = 'korean_cleaners2'
        symbols = symbols_ko

    elif language == 'ja':
        dataset_name = 'jsut'
        cleaners = 'japanese_cleaners2'
        symbols = symbols_ja

    elif language == 'en':
        dataset_name = 'ljspeech'
        cleaners = 'english_cleaners2'
        symbols = symbols_en

    else:
        print("Not supported language:"+language)
        return
    
    dataset_dir = f'../standard_dataset/{dataset_name}'
    
    
    open(f'../vits_dataset/{vc_model_name}/train.txt', 'w', encoding='utf-8').write(
        '\n'.join(open(f'{dataset_dir}/train.txt', 'r', encoding='utf-8').read(). \
                  replace(f'/{dataset_name}/', f'/{vc_model_name}/').split('\n'))
    )
    open(f'../vits_dataset/{vc_model_name}/val.txt', 'w', encoding='utf-8').write(
        '\n'.join(open(f'{dataset_dir}/val.txt', 'r', encoding='utf-8').read(). \
                  replace(f'/{dataset_name}/', f'/{vc_model_name}/').split('\n'))
    )
    open(f'../vits/text/symbols.py', 'w', encoding='utf-8'). \
        write(symbols_format.format(symbols))

    open(f'../vits/configs/{vc_model_name}.json', 'w', encoding='utf-8'). \
        write(vits_config_format.format(vc_model_name, vc_model_name, cleaners))

    os.chdir('../vits')
    import subprocess
    subprocess.Popen(f'python preprocess.py --text_cleaners {cleaners} --filelists ../vits_dataset/{vc_model_name}/train.txt ../vits_dataset/{vc_model_name}/val.txt'.split(' '))

    vc_multi(0, dataset_dir+'/wavs', f'../vits_dataset/{vc_model_name}/wavs', None, 0.0, 'crepe', '', '', 1, 7, 0, 1, 0.33, 'wav')    


vc_model_name = sys.argv[1]
make_dataset_language = sys.argv[2]
make_dataset(vc_model_name, make_dataset_language)