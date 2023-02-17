from utils.hparams import hparams
from .diffsinger_task import DiffSingerMIDITask, OpencpopDataset
from utils.pitch_utils import denorm_f0
import utils

class NaiveTask(DiffSingerMIDITask):
    def __init__(self):
        super(NaiveTask, self).__init__()
        self.dataset_cls = OpencpopDataset

    def run_model(self, model, sample, return_output=False, infer=False):
        '''
            steps:
            1. run the full model, calc the main loss
            2. calculate loss for dur_predictor, pitch_predictor, energy_predictor
        '''
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph'] # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        energy = sample.get('energy')
        key_shift = sample.get('key_shift')

        if infer:
            if hparams['use_spk_id']:
                spk_embed = model.fs2.spk_embed(sample['spk_ids'])[:, None, :]
            elif hparams['use_spk_embed']:
                spk_embed = sample['spk_embed']
            else:
                spk_embed = None
            output = model(txt_tokens, mel2ph=mel2ph, spk_mix_embed=spk_embed,
                           ref_mels=target, f0=f0, uv=uv, energy=energy, key_shift=key_shift, infer=infer)
        else:
            spk_embed = sample.get('spk_ids') if hparams['use_spk_id'] else sample.get('spk_embed')
            output = model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed,
                           ref_mels=target, f0=f0, uv=uv, energy=energy, key_shift=key_shift, infer=infer)

        losses = {}
        if 'diff_loss' in output:
            losses['mel'] = output['diff_loss']
        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]

        target = sample['mels']  # [B, T_s, 80]
        energy = sample.get('energy')
        key_shift = sample.get('key_shift')
        # fs2_mel = sample['fs2_mels']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph = sample['mel2ph']
        f0 = sample['f0']

        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            if hparams['use_spk_id']:
                spk_embed = self.model.fs2.spk_embed(sample['spk_ids'])[:, None, :]
            elif hparams['use_spk_embed']:
                spk_embed = sample['spk_embed']
            else:
                spk_embed = None
            model_out = self.model(
                txt_tokens, spk_mix_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=None, energy=energy,
                key_shift=key_shift, ref_mels=None, pitch_midi=sample['pitch_midi'],
                midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'), infer=True
            )

            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                gt_f0 = self.pe(sample['mels'])['f0_denorm_pred']  # pe predict from GT mel
                pred_f0 = self.pe(model_out['mel_out'])['f0_denorm_pred']  # pe predict from Pred mel
            else:
                gt_f0 = denorm_f0(sample['f0'], sample['uv'], hparams)
                pred_f0 = gt_f0
            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], is_mel=True, gt_f0=gt_f0, f0=pred_f0)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'], name=f'diffmel_{batch_idx}')

        return outputs
