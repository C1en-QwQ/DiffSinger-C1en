import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTHONPATH'] = f'"{root_dir}"'
sys.path.insert(0, root_dir)

import argparse

import onnx
import onnxsim
import torch
import torch.nn as nn
from torch.nn import Linear, Embedding

from utils import load_ckpt
from utils.hparams import set_hparams
from utils.phoneme_utils import build_phoneme_list
from modules.commons.common_layers import Embedding
from modules.diffsinger_midi.fs2 import FS_ENCODERS
from modules.fastspeech.fs2 import FastSpeech2
from modules.fastspeech.tts_modules import LayerNorm
from utils.hparams import hparams
from utils.text_encoder import TokenTextEncoder


class DurationPredictor(nn.Module):
    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, padding='SAME'):
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [nn.Sequential(
                nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == 'SAME'
                                       else (kernel_size - 1, 0), 0),
                nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                nn.Dropout(dropout_rate)
            )]
        if hparams['dur_loss'] in ['mse', 'huber']:
            odims = 1
        elif hparams['dur_loss'] == 'mog':
            odims = 15
        elif hparams['dur_loss'] == 'crf':
            odims = 32
            from torchcrf import CRF
            self.crf = CRF(odims, batch_first=True)
        else:
            raise NotImplementedError()
        self.linear = nn.Linear(n_chans, odims)

    def out2dur(self, xs):
        if hparams['dur_loss'] in ['mse']:
            # NOTE: calculate in log domain
            xs = xs.squeeze(-1)  # (B, Tmax)
            dur = xs.exp() - self.offset
            # dur = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()  # avoid negative value
        elif hparams['dur_loss'] == 'mog':
            raise NotImplementedError()
        elif hparams['dur_loss'] == 'crf':
            dur = torch.LongTensor(self.crf.decode(xs)).cuda()
        else:
            raise NotImplementedError()
        return dur

    def forward(self, xs, x_masks):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        conv_masks = x_masks[:, None, :]
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
            xs = xs * conv_masks
        xs = self.linear(xs.transpose(1, -1))  # [B, T, C]
        xs = xs * x_masks[:, :, None]  # (B, T, C)
        return self.out2dur(xs)


class FastSpeech2MIDI(FastSpeech2):
    def __init__(self, dictionary, out_dims=None):
        super().__init__(dictionary, out_dims)
        del self.encoder

        self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, self.encoder_embed_tokens, self.dictionary)
        self.midi_embed = Embedding(300, self.hidden_size, self.padding_idx)
        self.midi_dur_layer = Linear(1, self.hidden_size)
        self.is_slur_embed = Embedding(2, self.hidden_size)

        del self.dur_predictor
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=predictor_hidden,
            n_layers=hparams['dur_predictor_layers'],
            dropout_rate=hparams['predictor_dropout'], padding=hparams['ffn_padding'],
            kernel_size=hparams['dur_predictor_kernel'])

    def forward(self, txt_tokens, mel2ph=None, spk_embed_id=None,
                ref_mels=None, f0=None, uv=None, energy=None, skip_decoder=False,
                spk_embed_dur_id=None, spk_embed_f0_id=None, infer=False, **kwargs):
        midi_embedding = self.midi_embed(kwargs['midi'])
        midi_dur_embedding = self.midi_dur_layer(kwargs['midi_dur'][:, :, None])
        slur_embedding = self.is_slur_embed(kwargs['is_slur'].long())

        encoder_out = self.encoder(txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding)

        src_nonpadding = (txt_tokens > 0).float()
        dur_inp = encoder_out * src_nonpadding[:, :, None]
        dur = self.dur_predictor(dur_inp, src_nonpadding)
        dur = dur * (hparams['hop_size'] / hparams['audio_sample_rate'])
        
        return dur


class ModuleWrapper(nn.Module):
    def __init__(self, model, name='model'):
        super().__init__()
        self.wrapped_name = name
        setattr(self, name, model)

    def forward(self, *args, **kwargs):
        return getattr(self, self.wrapped_name)(*args, **kwargs)


class FastSpeech2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = ModuleWrapper(model, name='fs2')

    def forward(self, tokens, midi, midi_dur, is_slur):
        return self.model(tokens, midi=midi, midi_dur=midi_dur, is_slur=is_slur)


def build_fs2_model(device):
    model = FastSpeech2MIDI(
        dictionary=TokenTextEncoder(vocab_list=build_phoneme_list())
    )
    model.eval()
    load_ckpt(model, hparams['work_dir'], 'model.fs2', strict=True)
    model.to(device)
    return model


def export(fs2_path):
    set_hparams(print_hparams=False)
    if not hparams.get('use_midi', True):
        raise NotImplementedError('Only checkpoints of MIDI mode are supported.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fs2 = FastSpeech2Wrapper(
        model=build_fs2_model(device)
    )

    with torch.no_grad():
        tokens = torch.tensor([[3]], dtype=torch.long, device=device)
        midi = torch.tensor([[69]], dtype=torch.long, device=device)
        midi_dur = torch.tensor([[1.]], dtype=torch.float32, device=device)
        is_slur = torch.tensor([[False]], dtype=torch.bool, device=device)
        print('Exporting FastSpeech2...')
        torch.onnx.export(
            fs2,
            (
                tokens,
                midi,
                midi_dur,
                is_slur
            ),
            fs2_path,
            input_names=[
                'tokens',
                'midi',
                'midi_dur',
                'is_slur'
            ],
            output_names=[
                'ph_dur'
            ],
            dynamic_axes={
                'tokens': {
                    1: 'n_tokens'
                },
                'midi': {
                    1: 'n_tokens'
                },
                'midi_dur': {
                    1: 'n_tokens'
                },
                'is_slur': {
                    1: 'n_tokens'
                }
            },
            opset_version=11
        )
        model = onnx.load(fs2_path)
        in_dims = model.graph.input[0].type.tensor_type.shape.dim
        out_dims = model.graph.output[0].type.tensor_type.shape.dim
        out_dims.remove(out_dims[0])
        out_dims.insert(0, in_dims[0])
        out_dims.remove(out_dims[1])
        out_dims.insert(1, in_dims[1])
        model, check = onnxsim.simplify(model, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model, fs2_path)
        print('PyTorch ONNX export finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export DiffSinger acoustic model to ONNX')
    parser.add_argument('--exp', type=str, required=True, help='Experiment to export')
    parser.add_argument('--target', required=False, type=str, help='Path of the target ONNX model')
    args = parser.parse_args()

    cwd = os.getcwd()
    if args.target:
        target = os.path.join(cwd, args.target)
    else:
        target = None
    os.chdir(root_dir)
    exp = args.exp
    sys.argv = [
        'inference/ds_cascade.py',
        '--config',
        f'checkpoints/{exp}/config.yaml',
        '--exp_name',
        exp
    ]

    fs2_model_path = f'onnx/assets/{exp}.rhythmizer.onnx' if not target else target
    export(fs2_path=fs2_model_path)

    os.chdir(cwd)
    if args.target:
        log_path = os.path.abspath(args.target)
    else:
        log_path = fs2_model_path
    print(f'| export \'model.fs2\' to \'{log_path}\'.')
