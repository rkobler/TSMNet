
from importlib.metadata import metadata
from geoopt.manifolds.sphere import Sphere
from omegaconf.dictconfig import DictConfig
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset, TensorDataset
from library.utils.moabb import CachedParadigm
from spdnets.manifolds import SymmetricPositiveDefinite
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import Stiefel
import numpy as np
from copy import deepcopy
import time
import torch
from typing import Iterator, Sequence, Tuple
from sklearn.model_selection import StratifiedKFold
from datasetio.eeg.moabb import CachableDatase
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings


class BufferDataset(torch.utils.data.Dataset):

    def __init__(self, items) -> None:
        super().__init__()
        self.items = items

    def __len__(self):
        return self.items[0].shape[0]

    def __getitem__(self, index):
        return [item[index] for item in self.items]


class DomainIndex(int):
    '''
        Place holder class to get an entire domain
    '''
    pass


class DomainDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 labels : torch.LongTensor,
                 domains : torch.LongTensor,
                 metadata : pd.DataFrame,
                 training : bool = True, 
                 dtype : torch.dtype = torch.double,
                 mask_indices : Sequence[int] = None):

        self.dtype = dtype
        self._training = training
        self._metadata = metadata
        self._mask_indices = mask_indices
        assert(len(metadata) == len(labels))
        assert(len(metadata) == len(domains))
        self._metadata = metadata
        self._domains = domains
        self._labels =  labels

    @property
    def features(self) -> torch.Tensor:
        return self.get_features(range(len(self))).to(dtype=self.dtype)

    @property
    def metadata(self) -> pd.DataFrame:
        return self._metadata

    @property
    def domains(self) -> torch.Tensor:
        return self._domains

    @property
    def labels(self) -> torch.Tensor:
        labels = self._labels.clone()
        if self._mask_indices is not None and self.training:
            labels[self._mask_indices] = -1
        return labels

    @property
    def training(self) -> bool:
        return self._training

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def ndim(self):
        raise NotImplementedError()

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def set_masked_labels(self, indices):
        self._mask_indices = indices

    def get_feature(self, index : int) -> torch.Tensor:
        raise NotImplementedError()

    def get_features(self, indices) -> torch.Tensor:
        raise NotImplementedError()

    def copy(self, deep=False):
        raise NotImplementedError() 

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        if isinstance(index, DomainIndex):
            # load the data of an entire domain
            indices = np.flatnonzero(self.domains.numpy() == index)
            features = self.get_features(indices)
            return [dict(x=features.to(dtype=self.dtype),d=self.domains[indices]), self.labels[indices]]
        else:
            feature = self.get_feature(index)
            return [dict(x=feature.to(dtype=self.dtype),d=self.domains[index]), self.labels[index]]


class CachedDomainDataset(DomainDataset):

    def __init__(self, features, **kwargs) -> None:
        super().__init__(**kwargs)
        assert(len(self) == len(features))
        self._features = features

    @property
    def shape(self):
        return self._features.shape

    @property
    def ndim(self):
        return self._features.ndim

    def get_feature(self, index : int) -> torch.Tensor:
        return self._features[index]

    def get_features(self, indices) -> torch.Tensor:
        return self._features[indices]

    def set_features(self, features) -> torch.Tensor:
        if isinstance(features, np.ndarray):
            self._features = torch.from_numpy(features)
        elif isinstance(features, torch.Tensor):
            self._features = features
        else:
            raise ValueError()

    def copy(self, deep=False):
        
        features = self._features.clone() if deep else self._features
        labels = self._labels.clone() if deep else self._labels
        domains = self._domains.clone() if deep else self._domains

        obj = CachedDomainDataset(features, labels=labels, domains=domains, 
                                  metadata=self._metadata.copy(deep=deep), 
                                  training=self.training, dtype=self.dtype, 
                                  mask_indices=self._mask_indices)
        return obj


class CombinedDomainDataset(DomainDataset, torch.utils.data.ConcatDataset):

    def __init__(self, features : Sequence[torch.Tensor], **kwargs):

        torch.utils.data.ConcatDataset.__init__(self, features)
        DomainDataset.__init__(self, **kwargs)

    @classmethod
    def from_moabb(cls, paradigm : CachedParadigm, ds : CachableDatase, 
                   subjects : list = None, domain_expression = "session + subject * 1000", sessions : DictConfig = None,
                   **kwargs):
        if subjects is None:
            subjects = ds.subject_list
        features = []
        metadata = []
        labels = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for ix, subject in enumerate(subjects):
                x, l, md = paradigm.get_data(ds, [subject], False)
                
                if sessions is not None:
                    unique_sessions = md.session.unique()
                    if 'order' in sessions and sessions['order']== 'last':
                        unique_sessions = unique_sessions[::-1]
                    msk = md.session.isin(unique_sessions[0:sessions.get('n', len(unique_sessions))])
                    x = x[msk]
                    l = l[msk]
                    md = md[msk]

                features += [torch.from_numpy(x)]
                md['setindex'] = ix
                metadata += [md]
                labels += [l]
        metadata = pd.concat(metadata, ignore_index=True)
        labels =  torch.from_numpy(LabelEncoder().fit_transform(np.concatenate(labels))).to(dtype=torch.long)
        domains = torch.from_numpy(metadata.eval(domain_expression).to_numpy(dtype=np.int64))

        return CombinedDomainDataset(features=features, labels=labels, domains=domains, metadata=metadata, **kwargs)

    @property
    def shape(self):
        shape = list(self.datasets[0].shape)
        shape[0] = len(self)
        return tuple(shape)

    @property
    def ndim(self):
        return self.datasets[0].ndim

    def get_feature(self, index : int) -> torch.Tensor:
        return torch.utils.data.ConcatDataset.__getitem__(self, index)

    def get_features(self, indices) -> torch.Tensor:
        setix = self.metadata.loc[indices, 'setindex'].unique()
        if len(setix) > 1:
            raise ValueError('Domain data has to be contained in a single subset!')
        setix = setix[0]

        if setix == 0:
            subindices = indices
        else:
            subindices = indices - self.cumulative_sizes[setix - 1]

        return self.datasets[setix][subindices]

    def cache(self) -> CachedDomainDataset:
        features = torch.cat([ds.to(dtype=self.dtype) for ds in self.datasets])
        obj = CachedDomainDataset(features, labels=self.labels, domains=self._domains, metadata=self.metadata, 
                                  training = self.training, dtype=self.dtype,  
                                  mask_indices = self._mask_indices)
        return obj

    def copy(self, deep=False):
        features = [dataset.clone() if deep else dataset for dataset in self.datasets]
        obj = CombinedDomainDataset(features, labels=self.labels, domains=self._domains, metadata=self.metadata, 
                                    training = self.training, dtype=self.dtype, 
                                    mask_indices = self._mask_indices)
        return obj


class StratifyableDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, stratvar) -> None:
        super().__init__()
        self.dataset = dataset
        self.stratvar = stratvar
        assert(self.stratvar.shape[0] == len(dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]   

class BalancedDomainDataLoader(DataLoader):

    def __init__(self, dataset = None, batch_size = 1, domains_per_batch = 1, shuffle=True, replacement=False, **kwargs):
        if isinstance(dataset, Subset) and isinstance(dataset.dataset, DomainDataset):
            domains = dataset.dataset.domains[dataset.indices]
        elif isinstance(dataset, DomainDataset):
            domains = dataset.domains
        else:
            raise NotImplementedError()
        sampler = BalancedDomainSampler(domains, 
                int(batch_size/domains_per_batch), 
                shuffle=shuffle, replacement=replacement)
        super().__init__(dataset=dataset, sampler=sampler, batch_size=batch_size, **kwargs)


class StratifiedDataLoader(DataLoader):

    def __init__(self, dataset = None, batch_size = 1, shuffle=True, **kwargs):
        if isinstance(dataset, Subset) and isinstance(dataset.dataset, StratifyableDataset):
            stratvar = dataset.dataset.stratvar[dataset.indices]
        elif isinstance(dataset, StratifyableDataset):
            stratvar = dataset.stratvar
        else:
            raise NotImplementedError()

        sampler = StratifiedSampler(stratvar=stratvar, batch_size=batch_size, shuffle=shuffle)
        super().__init__(dataset=dataset, sampler=sampler, batch_size=batch_size, **kwargs)


class StratifiedDomainDataLoader(DataLoader):

    def __init__(self, dataset = None, batch_size = 1, domains_per_batch = 1, shuffle=True, **kwargs):

        if isinstance(dataset, Subset) and isinstance(dataset.dataset, Subset) and isinstance(dataset.dataset.dataset, (DomainDataset, CachedDomainDataset)):
            domains = dataset.dataset.dataset.domains[dataset.dataset.indices][dataset.indices]
            labels = dataset.dataset.dataset.domains[dataset.dataset.indices][dataset.indices]
        elif isinstance(dataset, Subset) and isinstance(dataset.dataset, (DomainDataset, CachedDomainDataset)):
            domains = dataset.dataset.domains[dataset.indices]
            labels = dataset.dataset.domains[dataset.indices]
        elif isinstance(dataset, (DomainDataset, CachedDomainDataset)):
            domains = dataset.domains
            labels = dataset.labels
        else:
            raise NotImplementedError()

        sampler = StratifiedDomainSampler(domains, labels,
                int(batch_size/domains_per_batch), domains_per_batch, 
                shuffle=shuffle)

        super().__init__(dataset=dataset, sampler=sampler, batch_size=batch_size, **kwargs)


def sample_without_replacement(domainlist, shuffle = True):
    du, counts = domainlist.unique(return_counts=True)
    dl = []
    while counts.sum() > 0:
        mask = counts > 0
        if shuffle:
            ixs = torch.randperm(du[mask].shape[0])
        else:
            ixs = range(du[mask].shape[0])
        counts[mask] -= 1
        dl.append(du[mask][ixs])
    return torch.cat(dl, dim=0)


class BalancedDomainSampler(torch.utils.data.Sampler[int]):
    def __init__(self, domains, samples_per_domain:int, shuffle = False, replacement = True) -> None:
        super().__init__(domains)
        self.samples_per_domain = samples_per_domain
        self.shuffle = shuffle
        self.replacement = replacement

        du, didxs, counts = domains.unique(return_inverse=True, return_counts=True)
        du = du.tolist()
        didxs = didxs.tolist()
        counts = counts.tolist()

        self.domainlist = torch.cat(
                [domain * torch.ones((counts[ix]//self.samples_per_domain), 
            dtype=torch.long) for ix, domain in enumerate(du)])
        
        self.domaindict = {}
        for domix, domid in enumerate(du):
            self.domaindict[domid] = torch.LongTensor(
                [idx for idx,dom in enumerate(didxs) if dom == domix])

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            if self.replacement:
                permidxs = torch.randperm(self.domainlist.shape[0])
                domainlist = self.domainlist[permidxs]
            else:
                domainlist = sample_without_replacement(self.domainlist, shuffle=True)
        else:
            if self.replacement:
                domainlist = self.domainlist
            else:
                domainlist = sample_without_replacement(self.domainlist, shuffle=False)

        generators = {}
        for domain in self.domaindict.keys():
            if self.shuffle:
                permidxs = torch.randperm(self.domaindict[domain].shape[0])
            else:
                permidxs = range(self.domaindict[domain].shape[0])
            generators[domain] = iter(
                torch.utils.data.BatchSampler(
                    self.domaindict[domain][permidxs].tolist(), 
                    batch_size=self.samples_per_domain, drop_last=True))

        for item in domainlist.tolist():
            batch = next(generators[item])
            yield from batch

    def __len__(self) -> int:
        return len(self.domainlist) * self.samples_per_domain


class StratifiedDomainSampler():

    def __init__(self, domains, stratvar, samples_per_domain, domains_per_batch, shuffle = True) -> None:
        self.samples_per_domain = samples_per_domain
        self.domains_per_batch = domains_per_batch
        self.shuffle = shuffle
        self.stratvar = stratvar

        du, didxs, counts = domains.unique(return_inverse=True, return_counts=True)
        du = du.tolist()
        didxs = didxs.tolist()

        self.domaincounts = torch.LongTensor((counts/self.samples_per_domain).tolist())
        
        self.domaindict = {}
        for domix, _ in enumerate(du):
            self.domaindict[domix] = torch.LongTensor(
                [idx for idx,dom in enumerate(didxs) if dom == domix])

    def __iter__(self) -> Iterator[int]:

        domaincounts = self.domaincounts.clone()

        generators = {}
        for domain in self.domaindict.keys():
            if self.shuffle:
                permidxs = torch.randperm(self.domaindict[domain].shape[0])
            else:
                permidxs = range(self.domaindict[domain].shape[0])
            generators[domain] = \
                iter(
                    StratifiedSampler(
                        self.stratvar[self.domaindict[domain]], 
                        batch_size=self.samples_per_domain,
                        shuffle=self.shuffle
                    ))
                # torch.utils.data.BatchSampler(
                #     self.domaindict[domain][permidxs].tolist(), 
                #     batch_size=self.samples_per_domain, drop_last=True))


        while domaincounts.sum() > 0:

            assert((domaincounts >= 0).all())
            # candidates = [ix for ix, count in enumerate(domaincounts.tolist()) if count > 0]
            candidates = torch.nonzero(domaincounts, as_tuple=False).flatten()
            if candidates.shape[0] < self.domains_per_batch:
                break

            # candidates = torch.LongTensor(candidates)
            permidxs = torch.randperm(candidates.shape[0])
            candidates = candidates[permidxs]

            # icap = min(len(candidates), self.domains_per_batch)
            batchdomains = candidates[:self.domains_per_batch]
            
            for item in batchdomains.tolist():
                within_domain_idxs = [next(generators[item]) for _ in range(self.samples_per_domain)]
                batch = self.domaindict[item][within_domain_idxs]
                # batch = next(generators[item])
                domaincounts[item] = domaincounts[item] - 1
                yield from batch
        yield from []

    def __len__(self) -> int:
        return self.domaincounts.sum() * self.samples_per_domain 


class StratifiedSampler(torch.utils.data.Sampler[int]):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, stratvar, batch_size, shuffle = True):
        self.n_splits = max(int(stratvar.shape[0] / batch_size), 2)
        self.stratvar = stratvar
        self.shuffle = shuffle

    def gen_sample_array(self):
        s = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle)   
        indices = [test for _, test in s.split(self.stratvar, self.stratvar)]
        return np.hstack(indices)

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.stratvar)
