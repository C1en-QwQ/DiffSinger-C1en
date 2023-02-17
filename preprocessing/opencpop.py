'''
    file -> temporary_dict -> processed_input -> batch
'''
import os
import traceback

import numpy as np
import torch
from librosa import note_to_midi

import utils
from basics.base_binarizer import BinarizationError
from data_gen.data_gen_utils import get_pitch_parselmouth
from src.vocoders.base_vocoder import VOCODERS
from tts.data_gen.txt_processors.zh_g2pM import get_all_vowels
from utils.hparams import hparams

vowels = get_all_vowels()


class File2Batch:
    '''
        pipeline: file -> temporary_dict -> processed_input -> batch
    '''

    @staticmethod
    def file2temporary_dict(raw_data_dir, ds_id):
        '''
            read from file, store data in temporary dicts
        '''
        # meta_midi = json.load(open(os.path.join(raw_data_dir, 'meta.json')))   # [list of dict]
        utterance_labels = open(os.path.join(raw_data_dir, 'transcriptions.txt'), encoding='utf-8').readlines()

        all_temp_dict = {}
        for utterance_label in utterance_labels:
            song_info = utterance_label.split('|')
            item_name = song_info[0]
            temp_dict = {
                'wav_fn': f'{raw_data_dir}/wavs/{item_name}.wav',
                'txt': song_info[1],
                'ph': song_info[2],
                'word_boundary': np.array([1 if x in vowels + ['AP', 'SP'] else 0 for x in song_info[2].split()]),
                'ph_durs': [float(x) for x in song_info[5].split(" ")],
                'pitch_midi': np.array([note_to_midi(x.split("/")[0]) if x != 'rest' else 0
                                        for x in song_info[3].split(" ")]),
                'midi_dur': np.array([float(x) for x in song_info[4].split(" ")]),
                'is_slur': np.array([int(x) for x in song_info[6].split(" ")]),
                'spk_id': ds_id
            }

            assert temp_dict['pitch_midi'].shape == temp_dict['midi_dur'].shape == temp_dict['is_slur'].shape, \
                (temp_dict['pitch_midi'].shape, temp_dict['midi_dur'].shape, temp_dict['is_slur'].shape)

            all_temp_dict[f'{ds_id}:{item_name}'] = temp_dict

        return all_temp_dict

    @staticmethod
    def temporary_dict2processed_input(item_name, temp_dict, encoder, binarization_args):
        '''
            process data in temporary_dicts
        '''

        def get_pitch(wav, mel):
            # get ground truth f0 by self.get_pitch_algorithm
            f0_path = f"{temp_dict['wav_fn'][:-4]}_f0.npy"
            if os.path.exists(f0_path):
                from utils.pitch_utils import f0_to_coarse
                processed_input['f0'] = np.load(f0_path)
                processed_input['pitch'] = f0_to_coarse(np.load(f0_path))
            else:
                gt_f0, gt_pitch_coarse = get_pitch_parselmouth(wav, mel, hparams)
                if sum(gt_f0) == 0:
                    raise BinarizationError("Empty **gt** f0")
                processed_input['f0'] = gt_f0
                processed_input['pitch'] = gt_pitch_coarse

        def get_align(meta_data, mel, phone_encoded, hop_size=hparams['hop_size'],
                      audio_sample_rate=hparams['audio_sample_rate']):
            mel2ph = np.zeros([mel.shape[0]], int)
            startTime = 0
            ph_durs = meta_data['ph_durs']

            for i_ph in range(len(ph_durs)):
                start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
                end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
                mel2ph[start_frame:end_frame] = i_ph + 1
                startTime = startTime + ph_durs[i_ph]

            processed_input['mel2ph'] = mel2ph

        mel_path = f"{temp_dict['wav_fn'][:-4]}_mel.npy"
        if os.path.exists(mel_path):
            wav = None
            mel = np.load(mel_path)
            print("load mel from npy")
        else:
            if hparams['vocoder'] in VOCODERS:
                wav, mel = VOCODERS[hparams['vocoder']].wav2spec(temp_dict['wav_fn'])
            else:
                wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(temp_dict['wav_fn'])
        processed_input = {
            'item_name': item_name, 'mel': mel, 'wav': wav,
            'sec': len(mel) * hparams["hop_size"] / hparams["audio_sample_rate"], 'len': mel.shape[0]
        }
        processed_input = {**temp_dict, **processed_input}  # merge two dicts
        try:
            if binarization_args['with_f0']:
                get_pitch(wav, mel)
            if binarization_args['with_txt']:
                try:
                    phone_encoded = processed_input['phone'] = encoder.encode(temp_dict['ph'])
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    get_align(temp_dict, mel, phone_encoded)
            if hparams.get('use_key_shift_embed', False):
                processed_input['key_shift'] = 0.
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {temp_dict['wav_fn']}")
            return None
        return processed_input

    @staticmethod
    def processed_input2batch(samples):
        '''
            Args:
                samples: one batch of processed_input
            NOTE:
                the batch size is controlled by hparams['max_sentences']
        '''
        if len(samples) == 0:
            return {}
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = utils.collate_1d([s['txt_token'] for s in samples], 0)
        f0 = utils.collate_1d([s['f0'] for s in samples], 0.0)
        pitch = utils.collate_1d([s['pitch'] for s in samples])
        uv = utils.collate_1d([s['uv'] for s in samples])
        mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], 0.0) \
            if samples[0]['mel2ph'] is not None else None
        mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'mels': mels,
            'mel_lengths': mel_lengths,
            'mel2ph': mel2ph,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        }
        if hparams['use_energy_embed']:
            batch['energy'] = utils.collate_1d([s['energy'] for s in samples], 0.0)
        if hparams.get('use_key_shift_embed', False):
            batch['key_shift'] = torch.FloatTensor([s['key_shift'] for s in samples])
        if hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        if hparams['pitch_type'] == 'cwt':
            cwt_spec = utils.collate_2d([s['cwt_spec'] for s in samples])
            f0_mean = torch.Tensor([s['f0_mean'] for s in samples])
            f0_std = torch.Tensor([s['f0_std'] for s in samples])
            batch.update({'cwt_spec': cwt_spec, 'f0_mean': f0_mean, 'f0_std': f0_std})
        elif hparams['pitch_type'] == 'ph':
            batch['f0'] = utils.collate_1d([s['f0_ph'] for s in samples])

        batch['pitch_midi'] = utils.collate_1d([s['pitch_midi'] for s in samples], 0)
        batch['midi_dur'] = utils.collate_1d([s['midi_dur'] for s in samples], 0)
        batch['is_slur'] = utils.collate_1d([s['is_slur'] for s in samples], 0)
        batch['word_boundary'] = utils.collate_1d([s['word_boundary'] for s in samples], 0)

        return batch
