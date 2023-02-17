"""
    item: one piece of data
    item_name: data id
    wavfn: wave file path
    txt: lyrics
    ph: phoneme
    tgfn: text grid file path (unused)
    spk: dataset name
    wdb: word boundary
    ph_durs: phoneme durations
    midi: pitch as midi notes
    midi_dur: midi duration
    is_slur: keep singing upon note changes
"""

import logging
import os.path
from copy import deepcopy

import matplotlib.pyplot as plt

from basics.base_binarizer import BaseBinarizer, BASE_ITEM_ATTRIBUTES
from utils.hparams import hparams
from utils.phoneme_utils import build_phoneme_list

ACOUSTIC_ITEM_ATTRIBUTES = BASE_ITEM_ATTRIBUTES + \
                           ['f0_fn', 'pitch_midi', 'midi_dur', 'is_slur', 'ph_durs', 'word_boundary']

class AcousticBinarizer(BaseBinarizer):
    def __init__(self):
        super().__init__(item_attributes=ACOUSTIC_ITEM_ATTRIBUTES)
        self.item_names = sorted(list(self.items.keys()))
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    def load_meta_data(self, raw_data_dir, ds_id):
        from preprocessing.opencpop import File2Batch
        self.items.update(File2Batch.file2temporary_dict(raw_data_dir, ds_id))
    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names

    @property
    def test_item_names(self):
        return self._test_item_names

    def get_align(self, meta_data, mel, phone_encoded, res):
        raise NotImplementedError()

    def split_train_test_set(self, item_names):
        item_names = set(deepcopy(item_names))
        prefixes = set([str(pr) for pr in hparams['test_prefixes']])
        test_item_names = set()
        # Add prefixes that specified speaker index and matches exactly item name to test set
        for prefix in deepcopy(prefixes):
            if prefix in item_names:
                test_item_names.add(prefix)
                prefixes.remove(prefix)
        # Add prefixes that exactly matches item name without speaker id to test set
        for prefix in deepcopy(prefixes):
            for name in item_names:
                if name.split(':')[-1] == prefix:
                    test_item_names.add(name)
                    prefixes.remove(prefix)
        # Add names with one of the remaining prefixes to test set
        for prefix in deepcopy(prefixes):
            for name in item_names:
                if name.startswith(prefix):
                    test_item_names.add(name)
                    prefixes.remove(prefix)
        for prefix in prefixes:
            for name in item_names:
                if name.split(':')[-1].startswith(prefix):
                    test_item_names.add(name)
        test_item_names = sorted(list(test_item_names))
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    def generate_summary(self, phone_set: set):
        # Group by phonemes.
        phoneme_map = {}
        for ph in sorted(phone_set):
            phoneme_map[ph] = 0
        if hparams['use_midi']:
            for item in self.items.values():
                for ph, slur in zip(item['ph'].split(), item['is_slur']):
                    if ph not in phone_set or slur == 1:
                        continue
                    phoneme_map[ph] += 1
        else:
            for item in self.items.values():
                for ph in item['ph'].split():
                    if ph not in phone_set:
                        continue
                    phoneme_map[ph] += 1

        print('===== Phoneme Distribution Summary =====')
        for i, key in enumerate(sorted(phoneme_map.keys())):
            if i == len(phone_set) - 1:
                end = '\n'
            elif i % 10 == 9:
                end = ',\n'
            else:
                end = ', '
            print(f'\'{key}\': {phoneme_map[key]}', end=end)

        # Draw graph.
        plt.figure(figsize=(int(len(phone_set) * 0.8), 10))
        x = list(phoneme_map.keys())
        values = list(phoneme_map.values())
        plt.bar(x=x, height=values)
        plt.tick_params(labelsize=15)
        plt.xlim(-1, len(phone_set))
        for a, b in zip(x, values):
            plt.text(a, b, b, ha='center', va='bottom', fontsize=15)
        plt.grid()
        plt.title('Phoneme Distribution Summary', fontsize=30)
        plt.xlabel('Phoneme', fontsize=20)
        plt.ylabel('Number of occurrences', fontsize=20)
        filename = os.path.join(hparams['binary_data_dir'], 'phoneme_distribution.jpg')
        plt.savefig(fname=filename,
                    bbox_inches='tight',
                    pad_inches=0.25)
        print(f'| save summary to \'{filename}\'')

    def load_ph_set(self, ph_set):
        # load those phones that appear in the actual data
        for item in self.items.values():
            ph_set += item['ph'].split(' ')
        # check unrecognizable or missing phones
        actual_phone_set = set(ph_set)
        required_phone_set = set(build_phoneme_list())
        self.generate_summary(required_phone_set)
        if actual_phone_set != required_phone_set:
            unrecognizable_phones = actual_phone_set.difference(required_phone_set)
            missing_phones = required_phone_set.difference(actual_phone_set)
            raise AssertionError('transcriptions and dictionary mismatch.\n'
                                 f' (+) {sorted(unrecognizable_phones)}\n'
                                 f' (-) {sorted(missing_phones)}')
