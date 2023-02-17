# coding=utf8
import argparse
import os
import sys

import numpy as np
import torch
import tqdm

from basics.base_svs_infer import BaseSVSInfer
from utils.infer_utils import cross_fade
from utils.audio import save_wav
from utils.hparams import set_hparams, hparams

sys.path.insert(0, '/')
root_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PYTHONPATH'] = f'"{root_dir}"'

parser = argparse.ArgumentParser(description='运行 DiffSinger 声码器')
parser.add_argument('mel', type=str, help='输入文件的路径')
parser.add_argument('--exp', type=str, required=False, help='从所选实验中读取声码器类和路径')
parser.add_argument('--config', type=str, required=False, help='从配置文件中读取声码器类和路径')
parser.add_argument('--class', type=str, required=False, help='指定声码器类')
parser.add_argument('--ckpt', type=str, required=False, help='指定声码器检查点路径')
parser.add_argument('--out', type=str, required=False, help='输出文件夹的路径')
parser.add_argument('--title', type=str, required=False, help='输出文件的标题')
args = parser.parse_args()

name = os.path.basename(args.mel).split('.')[0] if not args.title else args.title
config = None
if args.exp:
    config = f'{root_dir}/checkpoints/{args.exp}/config.yaml'
elif args.config:
    config = args.config
else:
    assert False, '应指定任一参数 \'--exp\' or \'--config\' .'

sys.argv = [
    f'{root_dir}/inference/ds_e2e.py',
    '--config',
    config
]
set_hparams(print_hparams=False)

cls = getattr(args, 'class')
if cls:
    hparams['vocoder'] = cls
if args.ckpt:
    hparams['vocoder_ckpt'] = args.ckpt


out = args.out
if not out:
    out = os.path.dirname(os.path.abspath(args.mel))

mel_seq = torch.load(args.mel)
sample_rate = hparams['audio_sample_rate']

infer_ins = None
if len(mel_seq) > 0:
    infer_ins = BaseSVSInfer(hparams, load_model=False)


def run_vocoder(path: str):
    result = np.zeros(0)
    current_length = 0

    for seg_mel in tqdm.tqdm(mel_seq, desc='mel segment', total=len(mel_seq)):
        seg_audio = infer_ins.run_vocoder(seg_mel['mel'].to(infer_ins.device), f0=seg_mel['f0'].to(infer_ins.device))
        seg_audio = seg_audio.squeeze(0).cpu().numpy()
        silent_length = round(seg_mel['offset'] * sample_rate) - current_length
        if silent_length >= 0:
            result = np.append(result, np.zeros(silent_length))
            result = np.append(result, seg_audio)
        else:
            result = cross_fade(result, seg_audio, current_length + silent_length)
        current_length = current_length + silent_length + seg_audio.shape[0]

    print(f'| save audio: {path}')
    save_wav(result, path, sample_rate)


os.makedirs(out, exist_ok=True)
run_vocoder(os.path.join(out, f'{name}.wav'))
