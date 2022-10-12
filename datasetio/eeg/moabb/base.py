from typing import Optional
import mne
import json

from moabb.datasets.base import BaseDataset

class CachableDatase(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return json.dumps({self.__class__.__name__: self.__dict__})

class PreprocessedDataset(CachableDatase):

    def __init__(self, *args, channels : Optional[list] = None, srate : Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.srate = srate

    def preprocess(self, raw):
        
        # find the events, first check stim_channels
        if len(mne.pick_types(raw.info, stim=True)) > 0:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
        else:
            events = None # the dataset already uses annotations

        # optional resampling
        if self.srate is not None:
            ret = raw.resample(self.srate, events=events)
            raw, events = (ret, events) if events is None else (ret[0], ret[1])

        # convert optional events to annotations (before we discard the stim channels)
        if events is not None:
            rev_event_it = dict(zip(self.event_id.values(), self.event_id.keys()))
            annot = mne.annotations_from_events(events, raw.info['sfreq'], event_desc=rev_event_it)
            raw.set_annotations(annot)

        # pick subset of all channels
        if self.channels is not None:
            raw.pick_channels(self.channels)
        else:
            raw.pick_types(eeg=True)

        return raw

    def __repr__(self) -> str:
        return json.dumps({self.__class__.__name__: self.__dict__})
