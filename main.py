# coding=utf8
import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

from utils.infer_utils import cross_fade, trans_key
from inference.ds_cascade import DiffSingerCascadeInfer
from inference.ds_e2e import DiffSingerE2EInfer
from utils.audio import save_wav
from utils.hparams import set_hparams, hparams
from utils.slur_utils import merge_slurs
from utils.spk_utils import parse_commandline_spk_mix

sys.path.insert(0, '/')
root_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PYTHONPATH'] = f'"{root_dir}"'

parser = argparse.ArgumentParser(description='运行DiffSinger推理')
parser.add_argument('proj', type=str, help='输入文件的路径')
parser.add_argument('--exp', type=str, required=True, help='型号的选择')
parser.add_argument('--spk', type=str, required=False, help='说话人名/多人名')
parser.add_argument('--out', type=str, required=False, help='输出文件夹的路径')
parser.add_argument('--title', type=str, required=False, help='输出文件的标题')
parser.add_argument('--num', type=int, required=False, default=1, help='运行次数')
parser.add_argument('--key', type=int, required=False, default=0, help='音高的关键转换')
parser.add_argument('--gender', type=float, required=False, help='共振峰移位（性控）')
parser.add_argument('--seed', type=int, required=False, help='推理的随机种子')
parser.add_argument('--speedup', type=int, required=False, default=0, help='PNDM 加速比')
parser.add_argument('--pitch', action='store_true', required=False, default=None, help='启用音高高低模式')
parser.add_argument('--forced_automatic_pitch_mode', action='store_true', required=False, default=False)
parser.add_argument('--mel', action='store_true', required=False, default=False,
                    help='保存中间 mel 格式而不是波形')
args = parser.parse_args()

# Deprecation for --pitch
warnings.filterwarnings(action='default')
if args.pitch is not None:
    warnings.warn(
        message='pitch \'--pitch\' 此参数已弃用,将来删除. '
                '程序现在会自动检测要使用的模式.',
        category=DeprecationWarning,
    )
    warnings.filterwarnings(action='default')

name = os.path.basename(args.proj).split('.')[0] if not args.title else args.title
exp = args.exp
if not os.path.exists(f'{root_dir}/checkpoints/{exp}'):
    for ckpt in os.listdir(os.path.join(root_dir, 'checkpoints')):
        if ckpt.startswith(exp):
            print(f'| 按前缀匹配 CKPT: {ckpt}')
            exp = ckpt
            break
    assert os.path.exists(f'{root_dir}/checkpoints/{exp}'), '匹配的exp没有在 \'checkpoints\' 文件夹中. ' \
                                                            '请具体说明 \'--exp\' 作为文件夹名称或前缀.'
else:
    print(f'| 按名称找到 CKPT: {exp}')

out = args.out
if not out:
    out = os.path.dirname(os.path.abspath(args.proj))

sys.argv = [
    f'{root_dir}/inference/ds_e2e.py' if not args.pitch else f'{root_dir}/inference/ds_cascade.py',
    '--exp_name',
    exp,
    '--infer'
]

if args.speedup > 0:
    sys.argv += ['--hparams', f'pndm_speedup={args.speedup}']

with open(args.proj, 'r', encoding='utf-8') as f:
    params = json.load(f)
if not isinstance(params, list):
    params = [params]

if args.key != 0:
    params = trans_key(params, args.key)
    key_suffix = '%+dkey' % args.key
    if not args.title:
        name += key_suffix
    print(f'音调基于原音频{key_suffix}')

if args.gender is not None:
    assert -1 <= args.gender <= 1, 'Gender must be in [-1, 1].'

set_hparams(print_hparams=False)
sample_rate = hparams['audio_sample_rate']

# Check for vocoder path
assert os.path.exists(os.path.join(root_dir, hparams['vocoder_ckpt'])), \
    f'声码器ckpt \'{hparams["vocoder_ckpt"]}\' 没有找到. ' \
    f'请将其放入检查点目录来跑推理.'

infer_ins = None
if len(params) > 0:
    if hparams['use_pitch_embed']:
        infer_ins = DiffSingerCascadeInfer(hparams, load_vocoder=not args.mel)
    else:
        warnings.warn(
            message='SVS MIDI-B version (implicit pitch prediction) is deprecated. '
            'Please select or train a model of MIDI-A version (controllable pitch prediction).',
            category=DeprecationWarning
        )
        warnings.filterwarnings(action='default')
        infer_ins = DiffSingerE2EInfer(hparams, load_vocoder=not args.mel)

spk_mix = parse_commandline_spk_mix(args.spk) if hparams['use_spk_id'] and args.spk is not None else None

for param in params:
    if args.gender is not None and hparams.get('use_key_shift_embed'):
        param['gender'] = args.gender
    if spk_mix is not None:
        param['spk_mix'] = spk_mix
    elif 'spk_mix' in param:
        param_spk_mix = param['spk_mix']
        for spk_name in param_spk_mix:
            values = str(param_spk_mix[spk_name]).split()
            if len(values) == 1:
                param_spk_mix[spk_name] = float(values[0])
            else:
                param_spk_mix[spk_name] = [float(v) for v in values]

    if not hparams.get('use_midi', False):
        merge_slurs(param)


def infer_once(path: str, save_mel=False):
    if save_mel:
        result = []
    else:
        result = np.zeros(0)
    current_length = 0

    for i, param in enumerate(params):
        # Ban automatic pitch mode by default
        param_have_f0 = 'f0_seq' in param and param['f0_seq']
        if hparams['use_pitch_embed'] and not param_have_f0:
            if not args.forced_automatic_pitch_mode:
                assert param_have_f0, '您使用的是自动升降声调模式，可能无法产生令人满意的效果 ' \
                                      '结果。当您看到此消息时，很可能您忘记了 ' \
                                      '将f0序列输入文件中，此错误将报告 ' \
                                      '您应该双次检查 ' \
                                      '自动升降声调模式，请手动打开.'
            warnings.warn(
                message='您正在使用强制自动升降声调模式。由于此模式仅用于测试目的, '
                        '请注意，你必须清楚地知道你在做什么，并意识到结果 '
                        '可能不令人满意.',
                category=UserWarning
            )
            warnings.filterwarnings(action='default')
            param['f0_seq'] = None

        if 'seed' in param:
            print(f'| set seed: {param["seed"] & 0xffff_ffff}')
            torch.manual_seed(param["seed"] & 0xffff_ffff)
            torch.cuda.manual_seed_all(param["seed"] & 0xffff_ffff)
        elif args.seed:
            print(f'| set seed: {args.seed & 0xffff_ffff}')
            torch.manual_seed(args.seed & 0xffff_ffff)
            torch.cuda.manual_seed_all(args.seed & 0xffff_ffff)
        else:
            torch.manual_seed(torch.seed() & 0xffff_ffff)
            torch.cuda.manual_seed_all(torch.seed() & 0xffff_ffff)

        if save_mel:
            mel, f0 = infer_ins.infer_once(param, return_mel=True)
            result.append({
                'offset': param.get('offset', 0.),
                'mel': mel,
                'f0': f0
            })
        else:
            seg_audio = infer_ins.infer_once(param)
            silent_length = round(param.get('offset', 0) * sample_rate) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_audio)
            else:
                result = cross_fade(result, seg_audio, current_length + silent_length)
            current_length = current_length + silent_length + seg_audio.shape[0]
        sys.stdout.flush()
        print('| finish segment: %d/%d (%.2f%%)' % (i + 1, len(params), (i + 1) / len(params) * 100))

    if save_mel:
        print(f'| save mel: {path}')
        torch.save(result, path)
    else:
        print(f'| save audio: {path}')
        save_wav(result, path, sample_rate)


os.makedirs(out, exist_ok=True)
suffix = '.wav' if not args.mel else '.mel.pt'
if args.num == 1:
    infer_once(os.path.join(out, f'{name}{suffix}'), save_mel=args.mel)
else:
    for i in range(1, args.num + 1):
        infer_once(os.path.join(out, f'{name}-{str(i).zfill(3)}{suffix}'), save_mel=args.mel)
