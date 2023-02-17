from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse, denorm_f0, norm_f0
import torch
import torch.nn as nn
from torch.nn import functional as F
from modules.commons.common_layers import Embedding, Linear
from modules.fastspeech.tts_modules import FastspeechEncoder, mel2ph_to_dur

class Encoder(FastspeechEncoder):
    def forward_embedding(self, txt_tokens, dur_embed):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        x = x + dur_embed
        if hparams['use_pos_embed']:
            if hparams.get('rel_pos') is not None and hparams['rel_pos']:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(txt_tokens)
                x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, txt_tokens, dur_embed):
        """
        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [T x B x C]
        }
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).detach()
        x = self.forward_embedding(txt_tokens, dur_embed)  # [B, T, H]
        x = super(FastspeechEncoder, self).forward(x, encoder_padding_mask)
        return x

class ParameterEncoder(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.txt_embed = Embedding(len(dictionary), hparams['hidden_size'], dictionary.pad())
        self.dur_embed = Linear(1, hparams['hidden_size'])
        self.encoder = Encoder(self.txt_embed, hparams['hidden_size'], hparams['enc_layers'], hparams['enc_ffn_kernel_size'], num_heads=hparams['num_heads'])

        self.f0_embed_type = hparams.get('f0_embed_type', 'discrete')
        if self.f0_embed_type == 'discrete':
            self.pitch_embed = Embedding(300, hparams['hidden_size'], dictionary.pad())
        elif self.f0_embed_type == 'continuous':
            self.pitch_embed = Linear(1, hparams['hidden_size'])
        else:
            raise ValueError('f0_embed_type must be \'discrete\' or \'continuous\'.')

        if hparams.get('use_key_shift_embed', False):
            self.key_shift_embed = Linear(1, hparams['hidden_size'])

        if hparams['use_spk_id']:
            self.spk_embed = Embedding(hparams['num_spk'], hparams['hidden_size'])
    
    def forward(self, txt_tokens, mel2ph=None, spk_embed_id=None,
                ref_mels=None, f0=None, uv=None, energy=None, skip_decoder=False,
                spk_embed_dur_id=None, spk_embed_f0_id=None, infer=False, is_slur=None, **kwarg):
        B, T = txt_tokens.shape
        dur = mel2ph_to_dur(mel2ph, T).float()
        dur_embed = self.dur_embed(dur[:, :, None])
        encoder_out = self.encoder(txt_tokens, dur_embed)
        
        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)
        
        nframes = mel2ph.size(1)
        delta_l = nframes - f0.size(1)
        if delta_l > 0:
            f0 = torch.cat((f0,torch.FloatTensor([[x[-1]] * delta_l for x in f0]).to(f0.device)),1)
        f0 = f0[:,:nframes]
        
        pitch_padding = (mel2ph == 0)
        f0_denorm = denorm_f0(f0, uv, hparams, pitch_padding=pitch_padding)
        if self.f0_embed_type == 'discrete':
            pitch = f0_to_coarse(f0_denorm)
            pitch_embed = self.pitch_embed(pitch)
        else:
            f0_mel = (1 + f0_denorm / 700).log()
            pitch_embed = self.pitch_embed(f0_mel[:, :, None])

        if hparams.get('use_key_shift_embed', False):
            key_shift = kwarg['key_shift']
            if len(key_shift.shape) == 1:
                key_shift_embed = self.key_shift_embed(key_shift[:, None, None])
            else:
                delta_l = nframes - key_shift.size(1)
                if delta_l > 0:
                    key_shift = torch.cat((key_shift, torch.FloatTensor([[x[-1]] * delta_l for x in key_shift]).to(key_shift.device)), 1)
                key_shift = key_shift[:, :nframes]
                key_shift_embed = self.key_shift_embed(key_shift[:, :, None])
        else:
            key_shift_embed = 0
        
        if hparams['use_spk_id']:
            if infer:
                spk_embed = kwarg.get('spk_mix_embed')  # (1, t, 256)
                mix_frames = spk_embed.size(1)
                if mix_frames > nframes:
                    spk_embed = spk_embed[:, :nframes, :]
                elif mix_frames > 1:
                    spk_embed = torch.cat((spk_embed, spk_embed[:, -1:, :].repeat(1, nframes - mix_frames, 1)), dim=1)
            else:
                spk_embed = self.spk_embed(spk_embed_id)[:, None, :]
        else:
            spk_embed = 0

        ret = {'decoder_inp': decoder_inp + pitch_embed + key_shift_embed + spk_embed, 'f0_denorm': f0_denorm}
        return ret
