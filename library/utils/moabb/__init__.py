import logging
import re
import json
import os
import hashlib
import numpy as np
import pandas as pd
import mne

from sklearn.base import BaseEstimator
from moabb.paradigms.base import BaseParadigm
from moabb.paradigms.motor_imagery import FilterBankMotorImagery, MotorImagery
from mne import get_config, set_config
from mne.datasets.utils import _get_path
from mne.io import read_info, write_info
from skorch import dataset


log = logging.getLogger(__name__)

class CachedParadigm(BaseParadigm):

    def _get_string_rep(self, obj):
        if issubclass(type(obj), BaseEstimator):
            str_repr = repr(obj.get_params())
        else:
            str_repr = repr(obj)
        str_no_addresses = re.sub("0x[a-z0-9]*", "0x__", str_repr)
        return str_no_addresses.replace("\n", "")

    def _get_rep(self, dataset):
        return self._get_string_rep(dataset) + '\n' + self._get_string_rep(self)

    def _get_cache_dir(self, rep):
        if get_config("MNEDATASET_TMP_DIR") is None:
            set_config("MNEDATASET_TMP_DIR", os.path.join(os.path.expanduser("~"), "mne_data"))
        base_dir = _get_path(None, "MNEDATASET_TMP_DIR", "preprocessed")        

        digest = hashlib.sha1(rep.encode("utf8")).hexdigest()

        cache_dir = os.path.join(
            base_dir,
            "preprocessed",
            digest
        )
        return cache_dir


    def process_raw(self, raw, dataset, return_epochs=False):
        # get events id
        event_id = self.used_events(dataset)

        # find the events, first check stim_channels then annotations
        stim_channels = mne.utils._get_stim_channel(None, raw.info,
                                                    raise_error=False)
        if len(stim_channels) > 0:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
        else:
            events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

        # picks channels
        if self.channels is None:
            picks = mne.pick_types(raw.info, eeg=True, stim=False)
        else:
            picks = mne.pick_types(raw.info, stim=False, include=self.channels)

        # pick events, based on event_id
        try:
            events = mne.pick_events(events, include=list(event_id.values()))
        except RuntimeError:
            # skip raw if no event found
            return

        # get interval
        tmin = self.tmin + dataset.interval[0]
        if self.tmax is None:
            tmax = dataset.interval[1]
        else:
            tmax = self.tmax + dataset.interval[0]

        X = []
        for bandpass in self.filters:
            fmin, fmax = bandpass
            # filter data
            if fmin is None and fmax is None:
                raw_f = raw
            else:
                raw_f = raw.copy().filter(fmin, fmax, method='iir',
                                          picks=picks, verbose=False)
            # epoch data
            epochs = mne.Epochs(raw_f, events, event_id=event_id,
                                tmin=tmin, tmax=tmax, proj=False,
                                baseline=None, preload=True,
                                verbose=False, picks=picks,
                                event_repeated='drop',
                                on_missing='ignore')
            if self.resample is not None:
                epochs = epochs.resample(self.resample)
            # rescale to work with uV
            if return_epochs:
                X.append(epochs)
            else:
                X.append(dataset.unit_factor * epochs.get_data())

        inv_events = {k: v for v, k in event_id.items()}
        labels = np.array([inv_events[e] for e in epochs.events[:, -1]])

        # if only one band, return a 3D array, otherwise return a 4D
        if len(self.filters) == 1:
            X = X[0]
        else:
            X = np.array(X).transpose((1, 2, 3, 0))

        metadata = pd.DataFrame(index=range(len(labels)))
        return X, labels, metadata


    def get_data(self, dataset, subjects=None, return_epochs=False):
        
        if return_epochs:
            raise ValueError("Only return_epochs=False is supported.")

        rep = self._get_rep(dataset)
        cache_dir = self._get_cache_dir(rep)
        os.makedirs(cache_dir, exist_ok=True)

        X = [] if return_epochs else np.array([])
        labels = []
        metadata = pd.Series([])

        if subjects is None:
            subjects = dataset.subject_list

        if not os.path.isfile(os.path.join(cache_dir, 'repr.json')):
            with open(os.path.join(cache_dir, 'repr.json'), 'w+') as f:
                f.write(self._get_rep(dataset))

        for subject in subjects:
            if not os.path.isfile(os.path.join(cache_dir, f'{subject}.npy')):
                # compute
                x, lbs, meta = super().get_data(dataset, [subject], return_epochs)
                np.save(os.path.join(cache_dir, f'{subject}.npy'), x)
                meta['label'] = lbs
                meta.to_csv(os.path.join(cache_dir, f'{subject}.csv'), index=False)
                log.info(f'saved cached data in directory {cache_dir}')

            # load from cache
            log.info(f'loading cached data from directory {cache_dir}')
            x = np.load(os.path.join(cache_dir, f'{subject}.npy'), mmap_mode ='r')
            meta = pd.read_csv(os.path.join(cache_dir, f'{subject}.csv'))
            lbs = meta['label'].tolist()

            if return_epochs:
                X.append(x)
            else:
                X = np.append(X, x, axis=0) if len(X) else x
            labels = np.append(labels, lbs, axis=0)
            metadata = pd.concat([metadata, meta], ignore_index=True)

        return X, labels, metadata

    def get_info(self, dataset):
        # check if the info has been saved
        rep = self._get_rep(dataset)
        cache_dir = self._get_cache_dir(rep)
        os.makedirs(cache_dir, exist_ok=True)
        info_file = os.path.join(cache_dir, f'raw-info.fif')
        if not os.path.isfile(info_file):
            x, _, _ = super().get_data(dataset, [dataset.subject_list[0]], True)
            info = x.info
            write_info(info_file, info)
            log.info(f'saved cached info in directory {cache_dir}')
        else:
            log.info(f'loading cached info from directory {cache_dir}')
            info = read_info(info_file)
        return info

    def __repr__(self) -> str:
        return json.dumps({self.__class__.__name__: self.__dict__})


class CachedMotorImagery(CachedParadigm, MotorImagery):

    def __init__(self, **kwargs):
        n_classes = len(kwargs['events'])
        super().__init__(n_classes=n_classes, **kwargs)


class CachedFilterBankMotorImagery(CachedParadigm, FilterBankMotorImagery):

    def __init__(self, **kwargs):
        n_classes = len(kwargs['events'])
        super().__init__(n_classes=n_classes, **kwargs)    

