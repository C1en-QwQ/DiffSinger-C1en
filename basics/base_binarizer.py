import shutil
import os
os.environ["OMP_NUM_THREADS"] = "1"

from utils.multiprocess_utils import chunked_multiprocess_run
import random
import json
from resemblyzer import VoiceEncoder
from tqdm import tqdm
from data_gen.data_gen_utils import get_mel2ph, get_pitch_parselmouth, build_phone_encoder
from utils.hparams import set_hparams, hparams
from utils.phoneme_utils import build_phoneme_list
import numpy as np
from utils.indexed_datasets import IndexedDatasetBuilder


class BinarizationError(Exception):
    pass

BASE_ITEM_ATTRIBUTES = ['txt', 'ph', 'wav_fn', 'tg_fn', 'spk_id']

class BaseBinarizer:
    '''
        Base class for data processing.
        1. *process* and *process_data_split*:
            process entire data, generate the train-test split (support parallel processing);
        2. *process_item*:
            process singe piece of data;
        3. *get_pitch*:
            infer the pitch using some algorithm;
        4. *get_align*:
            get the alignment using 'mel2ph' format (see https://arxiv.org/abs/1905.09263).
        5. phoneme encoder, voice encoder, etc.

        Subclasses should define:
        1. *load_metadata*:
            how to read multiple datasets from files;
        2. *train_item_names*, *valid_item_names*, *test_item_names*:
            how to split the dataset;
        3. load_ph_set:
            the phoneme set.
    '''
    def __init__(self, data_dir=None, item_attributes=None):
        if item_attributes is None:
            item_attributes = BASE_ITEM_ATTRIBUTES
        if data_dir is None:
            data_dir = hparams['raw_data_dir']

        if 'speakers' not in hparams:
            speakers = hparams['datasets']
            hparams['speakers'] = hparams['datasets']
        else:
            speakers = hparams['speakers']
        assert isinstance(speakers, list), 'Speakers must be a list'
        assert len(speakers) == len(set(speakers)), 'Speakers cannot contain duplicate names'

        self.raw_data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        assert len(speakers) == len(self.raw_data_dirs), \
            'Number of raw data dirs must equal number of speaker names!'

        self.binarization_args = hparams['binarization_args']
        self.augmentation_args = hparams.get('augmentation_args', {})
        self.pre_align_args = hparams['pre_align_args']
        
        self.items = {}
        # every item in self.items has some attributes
        self.item_attributes = item_attributes

        # load each dataset
        for ds_id, data_dir in enumerate(self.raw_data_dirs):
            self.load_meta_data(data_dir, ds_id)
            if ds_id == 0:
                # check program correctness
                assert all([attr in self.item_attributes for attr in list(self.items.values())[0].keys()])
        self.item_names = sorted(list(self.items.keys()))
        
        if self.binarization_args['shuffle']:
            random.seed(hparams['seed'])
            random.shuffle(self.item_names)
        
        # set default get_pitch algorithm
        self.get_pitch_algorithm = get_pitch_parselmouth

    def load_meta_data(self, raw_data_dir, ds_id):
        raise NotImplementedError

    def split_train_test_set(self, item_names):
        raise NotImplementedError

    @property
    def train_item_names(self):
        raise NotImplementedError

    @property
    def valid_item_names(self):
        raise NotImplementedError

    @property
    def test_item_names(self):
        raise NotImplementedError

    def build_spk_map(self):
        spk_map = {x: i for i, x in enumerate(hparams['speakers'])}
        assert len(spk_map) <= hparams['num_spk'], 'Actual number of speakers should be smaller than num_spk!'
        return spk_map

    def item_name2spk_id(self, item_name):
        return self.spk_map[self.items[item_name]['spk_id']]

    def _phone_encoder(self):
        ph_set = []
        # Just for ensuring the transcriptions match the dictionary.
        # May need refactoring in the future.
        dict_fn = os.path.join(hparams['binary_data_dir'], 'dictionary.txt')
        if hparams['reset_phone_dict'] or not os.path.exists(dict_fn):
            self.load_ph_set(ph_set)  # For singing, do checking and return the correct results.
            ph_set = sorted(set(ph_set))
            shutil.copy(hparams['g2p_dictionary'], dict_fn)
        else:
            ph_set = build_phoneme_list()
        return build_phone_encoder(ph_set)

    def load_ph_set(self, ph_set):
        raise NotImplementedError

    def meta_data_iterator(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            meta_data = self.items[item_name]
            yield item_name, meta_data

    def process(self):
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.spk_map = self.build_spk_map()
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w', encoding='utf-8'))

        self.phone_encoder = self._phone_encoder()
        self.process_data_split('valid')
        self.process_data_split('test')
        self.process_data_split('train', apply_augmentation=len(self.augmentation_args) > 0)

    def process_data_split(self, prefix, multiprocess=False, apply_augmentation=False):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        lengths = []
        f0s = []
        total_sec = 0
        total_raw_sec = 0

        if self.binarization_args['with_spk_embed']:
            voice_encoder = VoiceEncoder().cuda()

        for item_name, meta_data in self.meta_data_iterator(prefix):
            args.append([item_name, meta_data, self.binarization_args])

        aug_map = self.arrange_data_augmentation(prefix) if apply_augmentation else {}

        def postprocess(item_):
            nonlocal total_sec, total_raw_sec
            if item_ is None:
                return
            item_['spk_embed'] = voice_encoder.embed_utterance(item_['wav']) \
                if self.binarization_args['with_spk_embed'] else None
            if not self.binarization_args['with_wav'] and 'wav' in item_:
                del item_['wav']
            builder.add_item(item_)
            lengths.append(item_['len'])
            total_sec += item_['sec']
            total_raw_sec += item_['sec']
            if item_.get('f0') is not None:
                f0s.append(item_['f0'])

            for task in aug_map.get(item_['item_name'], []):
                aug_item = task['func'](item_, **task['kwargs'])
                builder.add_item(aug_item)
                lengths.append(aug_item['len'])
                total_sec += aug_item['sec']
                if aug_item.get('f0') is not None:
                    f0s.append(aug_item['f0'])

        if multiprocess:
            # code for parallel processing
            num_workers = int(os.getenv('N_PROC', hparams.get('ds_workers', os.cpu_count() // 3)))
            for item in tqdm(
                chunked_multiprocess_run(self.process_item, args, num_workers=num_workers),
                total=len(list(self.meta_data_iterator(prefix)))
            ):
                postprocess(item)
        else:
            # code for single cpu processing
            for a in tqdm(args):
                item = self.process_item(*a)
                postprocess(item)
        
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])

        if apply_augmentation:
            print(f'| {prefix} total duration (before augmentation): {total_raw_sec:.2f}s')
            print(f'| {prefix} total duration (after augmentation): {total_sec:.2f}s ({total_sec / total_raw_sec:.2f}x)')
        else:
            print(f'| {prefix} total duration: {total_raw_sec:.2f}s')

    def arrange_data_augmentation(self, prefix):
        """
        Code for all types of data augmentation should be added here.
        """
        aug_map = {}
        all_item_names = [item_name for item_name, _ in self.meta_data_iterator(prefix)]
        if self.augmentation_args.get('random_pitch_shifting') is not None:
            from augmentation.pitch_shift import PitchShiftAugmentation
            aug_args = self.augmentation_args['random_pitch_shifting']
            key_shift_min, key_shift_max = aug_args['range']
            assert hparams.get('use_key_shift_embed', False), \
                'Random pitch shifting augmentation requires use_key_shift_embed == True.'
            assert key_shift_min < 0 < key_shift_max, \
                'Random pitch shifting augmentation must have a range where min < 0 < max.'

            aug_ins = PitchShiftAugmentation(self.raw_data_dirs, aug_args)
            scale = aug_args['scale']
            aug_item_names = all_item_names * int(scale) \
                             + random.sample(all_item_names, int(len(all_item_names) * (scale - int(scale))))

            for aug_item_name in aug_item_names:
                rand = random.random() * 2 - 1
                if rand < 0:
                    key_shift = key_shift_min * abs(rand)
                else:
                    key_shift = key_shift_max * rand
                aug_task = {
                    'func': aug_ins.process_item,
                    'kwargs': {'key_shift': key_shift}
                }
                if aug_item_name in aug_map:
                    aug_map[aug_item_name].append(aug_task)
                else:
                    aug_map[aug_item_name] = [aug_task]

        if self.augmentation_args.get('fixed_pitch_shifting') is not None:
            from augmentation.pitch_shift import PitchShiftAugmentation
            aug_args = self.augmentation_args['fixed_pitch_shifting']
            targets = aug_args['targets']
            scale = aug_args['scale']
            assert self.augmentation_args.get('random_pitch_shifting') is None, \
                'Fixed pitch shifting augmentation is not compatible with random pitch shifting.'
            assert len(targets) == len(set(targets)), \
                'Fixed pitch shifting augmentation requires having no duplicate targets.'
            assert hparams['use_spk_id'], 'Fixed pitch shifting augmentation requires use_spk_id == True.'
            assert hparams['num_spk'] >= (1 + len(targets)) * len(self.spk_map), \
                'Fixed pitch shifting augmentation requires num_spk >= (1 + len(targets)) * len(speakers).'
            assert scale < 1, 'Fixed pitch shifting augmentation requires scale < 1.'

            aug_ins = PitchShiftAugmentation(self.raw_data_dirs, aug_args)
            for i, target in enumerate(targets):
                aug_item_names = random.sample(all_item_names, int(len(all_item_names) * scale))
                for aug_item_name in aug_item_names:
                    replace_spk_id = int(aug_item_name.split(':', maxsplit=1)[0]) + (i + 1) * len(self.spk_map)
                    aug_task = {
                        'func': aug_ins.process_item,
                        'kwargs': {'key_shift': target, 'replace_spk_id': replace_spk_id}
                    }
                    if aug_item_name in aug_map:
                        aug_map[aug_item_name].append(aug_task)
                    else:
                        aug_map[aug_item_name] = [aug_task]

        return aug_map

    def process_item(self, item_name, meta_data, binarization_args):
        from preprocessing.opencpop import File2Batch
        return File2Batch.temporary_dict2processed_input(item_name, meta_data, self.phone_encoder, binarization_args)

    def get_align(self, meta_data, mel, phone_encoded, res):
        raise NotImplementedError

    def get_align_from_textgrid(self, meta_data, mel, phone_encoded, res):
        '''
            NOTE: this part of script is *isolated* from other scripts, which means
                  it may not be compatible with the current version.
        '''
        return
        tg_fn, ph = meta_data['tg_fn'], meta_data['ph']
        if tg_fn is not None and os.path.exists(tg_fn):
            mel2ph, dur = get_mel2ph(tg_fn, ph, mel, hparams)
        else:
            raise BinarizationError(f"Align not found")
        if mel2ph.max() - 1 >= len(phone_encoded):
            raise BinarizationError(
                f"Align does not match: mel2ph.max() - 1: {mel2ph.max() - 1}, len(phone_encoded): {len(phone_encoded)}")
        res['mel2ph'] = mel2ph
        res['dur'] = dur

    def get_f0cwt(self, f0, res):
        '''
            NOTE: this part of script is *isolated* from other scripts, which means
                  it may not be compatible with the current version.
        '''
        return
        from utils.cwt import get_cont_lf0, get_lf0_cwt
        uv, cont_lf0_lpf = get_cont_lf0(f0)
        logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
        cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)
        if np.any(np.isnan(Wavelet_lf0)):
            raise BinarizationError("NaN CWT")
        res['cwt_spec'] = Wavelet_lf0
        res['cwt_scales'] = scales
        res['f0_mean'] = logf0s_mean_org
        res['f0_std'] = logf0s_std_org


if __name__ == "__main__":
    set_hparams()
    BaseBinarizer().process()
