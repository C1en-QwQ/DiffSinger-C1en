from copy import deepcopy

from basics.base_augmentation import BaseAugmentation
from src.vocoders.base_vocoder import VOCODERS
from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse


class PitchShiftAugmentation(BaseAugmentation):
    def __init__(self, data_dirs: list, augmentation_args: dict):
        super().__init__(data_dirs, augmentation_args)


    def process_item(self, item: dict, key_shift=0, replace_spk_id=None) -> dict:
        aug_item = deepcopy(item)
        if hparams['vocoder'] in VOCODERS:
            _, mel = VOCODERS[hparams['vocoder']].wav2spec(aug_item['wav_fn'], keyshift=key_shift)
        else:
            _, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(aug_item['wav_fn'], keyshift=key_shift)
        aug_item['key_shift'] = key_shift
        aug_item['mel'] = mel
        aug_item['f0'] *= 2 ** (key_shift / 12)
        aug_item['pitch'] = f0_to_coarse(aug_item['f0'])
        if replace_spk_id is not None:
            aug_item['spk_id'] = replace_spk_id
        return aug_item
