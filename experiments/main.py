#%%
import os
from time import time
from hydra.core.hydra_config import HydraConfig
import pandas as pd
from skorch.callbacks.scoring import EpochScoring
from skorch.dataset import ValidSplit
from skorch.callbacks import Checkpoint

import logging
import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict

import moabb
from sklearn.metrics import get_scorer, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold
from library.utils.moabb import CachedParadigm
from spdnets.models import DomainAdaptBaseModel, DomainAdaptJointTrainableModel, EEGNetv4
from spdnets.models import CPUModel
# from library.utils.moabb.evaluation import extract_adapt_idxs
import warnings
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from library.utils.torch import BalancedDomainDataLoader, CombinedDomainDataset, DomainIndex, StratifiedDomainDataLoader
from spdnets.models.base import DomainAdaptFineTuneableModel, FineTuneableModel

from spdnets.utils.skorch import DomainAdaptNeuralNetClassifier
from library.utils.hydra import hydra_helpers


log = logging.getLogger(__name__)

rng_seed = 42

warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


@hydra_helpers
@hydra.main(config_path='conf/', config_name='mims-nnets.yaml')
def main(cfg : DictConfig):

    moabb.set_log_level("info")

    rootdir = hydra.utils.to_absolute_path(os.path.dirname(__file__))

    # print(OmegaConf.to_yaml(cfg))

    if torch.cuda.is_available():
        gpuid = f"cuda:{HydraConfig.get().job.get('num', 0) % torch.cuda.device_count()}"
        log.debug(f"GPU ID: {gpuid}")
        device = torch.device(gpuid)
    else:
        device = torch.device('cpu')
    cpu = torch.device('cpu')

    with open_dict(cfg):
        if 'ft_pipeline' not in cfg.nnet:
            cfg.nnet.ft_pipeline = None
        if 'prep_pipeline' not in cfg.nnet:
            cfg.nnet.prep_pipeline = None

    dataset = hydra.utils.instantiate(cfg.dataset.type,_convert_='partial')
    ppreprocessing_dict = hydra.utils.instantiate(cfg.preprocessing,_convert_='partial')
    assert(len(ppreprocessing_dict) == 1) # only 1 paradigm is allowed per call
    prep_name, paradigm = next(iter(ppreprocessing_dict.items()))

    res_dir = os.path.join(cfg.evaluation.strategy,prep_name)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    results = pd.DataFrame( \
        columns = ['dataset','subject','session','method', 'score_trn', 'score_tst', 
                   'time', 'n_test','classes'])
    resix = 0
    results['score_trn'] = results['score_trn'].astype(np.double)
    results['score_tst'] = results['score_tst'].astype(np.double)
    results['time'] = results['time'].astype(np.double)
    results['n_test'] = results['n_test'].astype(int)
    results['classes'] = results['classes'].astype(int)

    results_fit = []

    scorefun = get_scorer(cfg.score)._score_func

    def masked_scorefun(y_true, y_pred, **kwargs):
        masked = y_true == -1
        if np.all(masked):
            log.warning('Nothing to score because all target values are masked (value = -1).')
            return np.nan
        return scorefun(y_true[~masked], y_pred[~masked], **kwargs)

    scorer = make_scorer(masked_scorefun)
    
    dadapt = cfg.evaluation.adapt 

    bacc_val_logger = EpochScoring(scoring=scorer, 
                                lower_is_better=False, 
                                on_train=False,
                                name='score_val')
    bacc_trn_logger = EpochScoring(scoring=scorer, 
                    lower_is_better=False, 
                    on_train=True,
                    name='score_trn')

    if 'inter-session' in cfg.evaluation.strategy:
        subset_iter = iter([[s] for s in dataset.subject_list])
        groupvarname = 'session'
    elif 'inter-subject' in cfg.evaluation.strategy:
        subset_iter = iter([None])
        groupvarname = 'subject'
    else:
        raise NotImplementedError()

    for subset in subset_iter:

        if groupvarname == 'session':
            domain_expression = "session"
        elif groupvarname == 'subject':
            domain_expression = "session + subject * 1000"

        selected_sessions = cfg.dataset.get("sessions", None)

        ds = CombinedDomainDataset.from_moabb(paradigm, dataset, subjects=subset, domain_expression=domain_expression, dtype=cfg.nnet.inputtype, sessions=selected_sessions)

        if cfg.nnet.prep_pipeline is not None:
            ds = ds.cache() # we need to load the entire dataset for preprocessing

        sessions = ds.metadata.session.astype(np.int64).values
        subjects = ds.metadata.subject.astype(np.int64).values

        g = ds.metadata[groupvarname].astype(np.int64).values
        groups = np.unique(g)

        domains = ds.domains.unique()

        n_classes = len(ds.labels.unique())

        if len(groups) < 2:
            log.warning(f"Insufficient number (n={len(groups)}) of groups ({groupvarname}) in the (sub-)dataset to run leave 1 group out CV!")
            continue

        mdl_kwargs = dict(nclasses=n_classes)

        mdl_kwargs['nchannels'] = ds.shape[1]
        mdl_kwargs['nsamples'] = ds.shape[2]
        mdl_kwargs['nbands'] = ds.shape[3] if ds.ndim == 4 else 1
        mdl_kwargs['input_shape'] = (1, ) + ds.shape[1:]

        mdl_dict = OmegaConf.to_container(cfg.nnet.model, resolve=True)
        mdl_class = hydra.utils.get_class(mdl_dict.pop('_target_'))

        if issubclass(mdl_class, DomainAdaptBaseModel):
            mdl_kwargs['domains'] = domains
        if issubclass(mdl_class, EEGNetv4):
            if isinstance(paradigm, CachedParadigm):
                info = paradigm.get_info(dataset)
                mdl_kwargs['srate'] = int(info['sfreq'])
            else:
                raise NotImplementedError()
        if issubclass(mdl_class, FineTuneableModel) and isinstance(ds, CombinedDomainDataset):
            # we need to load the entire dataset
            ds = ds.cache()
        
        mdl_kwargs = {**mdl_kwargs, **mdl_dict}

        optim_kwargs = OmegaConf.to_container(cfg.nnet.optimizer, resolve=True)
        optim_class =  hydra.utils.get_class(optim_kwargs.pop('_target_'))

        metaddata = {
            'model_class' : mdl_class,
            'model_kwargs' : mdl_kwargs,
            'optim_class' : optim_class,
            'optim_kwargs' : optim_kwargs
        }

        mdl_metadata_dir = os.path.join(res_dir,'metadata')
        if not os.path.exists(mdl_metadata_dir):
            os.makedirs(mdl_metadata_dir)

        torch.save(metaddata, f=os.path.join(mdl_metadata_dir, f'meta-{cfg.nnet.name}.pth'))
        with open(os.path.join(mdl_metadata_dir, f'config-{cfg.nnet.name}.yaml'), 'w+') as f:
            f.writelines(OmegaConf.to_yaml(cfg))

        if issubclass(mdl_class, CPUModel):
            device = cpu

        mdl_kwargs['device'] = device

        n_test_groups = int(np.clip(np.round(len(groups) * cfg.fit.test_size), 1, None))

        log.info(f"Performing leave {n_test_groups} (={cfg.fit.test_size*100:.0f}%) {groupvarname}(s) out CV")
        cv = GroupKFold(n_splits=int(len(groups)/n_test_groups))

        ds.eval() # unmask labels
        for train, test in cv.split(ds.labels, ds.labels, g):

            target_domains = ds.domains[test].unique().numpy()
            torch.manual_seed(rng_seed + target_domains[0])

            prep_pipeline = hydra.utils.instantiate(cfg.nnet.prep_pipeline,_convert_='partial')
            ft_pipeline = hydra.utils.instantiate(cfg.nnet.ft_pipeline,_convert_='partial')

            if dadapt is not None and dadapt.name != 'no':
                # extend training data with adaptation set
                if issubclass(mdl_class, DomainAdaptJointTrainableModel):
                    stratvar = ds.labels + ds.domains * n_classes
                    adapt_domain = test #extract_adapt_idxs(dadapt.nadapt_domain, test, stratvar)
                else:
                    # some nets to not require target domain data during training
                    adapt_domain = np.array([], dtype=np.int64)
                    log.info("Model does not require adaptation. Using original training data.")

                train_source_doms = train
                train = np.concatenate((train, adapt_domain)) 

                if dadapt.name == 'uda':
                    ds.set_masked_labels(adapt_domain)
                elif dadapt.name == 'sda':
                    test = np.setdiff1d(test, adapt_domain)

                if len(test) == 0:
                    raise ValueError('No data left in the test set!')
            else:
                train_source_doms = train

            test_groups = np.unique(g[test])
            test_group_list = []
            for test_group in test_groups:
                test_dict = {}
                subject = np.unique(subjects[g == test_group])
                assert(len(subject) == 1) # only one subject per group
                test_dict['subject'] = subject[0]
                if groupvarname == 'subject':
                    test_dict['session'] = -1
                else:
                    session = np.unique(sessions[g == test_group])
                    assert(len(session) == 1) # only one session per group
                    test_dict['session'] = session[0]
                test_dict['idxs'] = np.intersect1d(test, np.nonzero(g == test_group))
                test_group_list.append(test_dict)

            t_start = time()

            ## preprocessing
            dsprep = ds.copy(deep=False)
            dsprep.train() # mask labels

            if prep_pipeline is not None:
                prep_pipeline.fit(dsprep.features[train].numpy(), dsprep.labels[train])
                dsprep.set_features(prep_pipeline.transform(dsprep.features))

            # torch dataset generation
            batch_size_valid = cfg.fit.validation_size if type(cfg.fit.validation_size) == int else int(np.ceil(cfg.fit.validation_size * len(train)))

            dsprep.eval() # unmask labels
            # extract stratified (classes and groups) validation data
            stratvar = dsprep.labels[train] + dsprep.domains[train] * n_classes
            valid_cv = ValidSplit(iter(StratifiedShuffleSplit(n_splits = 1, test_size=cfg.fit.validation_size, 
                                              random_state=rng_seed + target_domains[0]).split(stratvar, stratvar)))        

            netkwargs = {'module__'+k: v for k, v in mdl_kwargs.items()}
            netkwargs = {**netkwargs, **{'optimizer__'+k: v for k, v in optim_kwargs.items()}}
            if cfg.fit.stratified:

                n_train_domains = len(dsprep.domains[train].unique())
                domains_per_batch = min(cfg.fit.domains_per_batch, n_train_domains)
                batch_size_train = int(max(np.round(cfg.fit.batch_size_train/domains_per_batch), 2)*domains_per_batch)

                netkwargs['iterator_train'] = StratifiedDomainDataLoader
                netkwargs['iterator_train__domains_per_batch'] = domains_per_batch
                netkwargs['iterator_train__shuffle'] = True
                netkwargs['iterator_train__batch_size'] = batch_size_train
            else:
                netkwargs['iterator_train'] = BalancedDomainDataLoader
                netkwargs['iterator_train__domains_per_batch'] = cfg.fit.domains_per_batch
                netkwargs['iterator_train__drop_last'] = True
                netkwargs['iterator_train__replacement'] = False
                netkwargs['iterator_train__batch_size'] = cfg.fit.batch_size_train
            netkwargs['iterator_valid__batch_size'] = batch_size_valid
            netkwargs['max_epochs'] = cfg.fit.epochs
            netkwargs['callbacks__print_log__prefix'] = f'{dataset.code} {n_classes}cl | {test_groups} | {cfg.nnet.name} :'

            mdl_path_tmp = os.path.join(res_dir,'models','tmp',f'{test_groups}_{cfg.nnet.name}.pth')
            if not os.path.exists(os.path.split(mdl_path_tmp)[0]):
                os.makedirs(os.path.split(mdl_path_tmp)[0])
            checkpoint = Checkpoint(
                f_params=mdl_path_tmp, f_criterion=None,f_optimizer=None,f_history=None,
                monitor='valid_loss_best', load_best=True)

            scheduler = hydra.utils.instantiate(cfg.nnet.scheduler,_convert_='partial')

            net = DomainAdaptNeuralNetClassifier(
                mdl_class,
                train_split=valid_cv, 
                callbacks=[bacc_trn_logger,bacc_val_logger, scheduler, checkpoint],
                optimizer=optim_class,
                verbose=0,
                device=device,
                **netkwargs)

            dsprep.train() # mask labels
            dstrn = torch.utils.data.Subset(dsprep, train)
            net.fit(dstrn, None)
            
            res = pd.DataFrame(net.history)
            res = res.drop(res.filter(regex='.*batches|_best|_count').columns, axis=1)
            res = res.drop(res.filter(regex='event.*').columns, axis=1)
            res = res.rename(columns=dict(train_loss="loss_trn", valid_loss="loss_val", dur="time"))
            res['domains'] = str(test_groups)
            res['method'] = cfg.nnet.name
            res['dataset'] = dataset.code
            results_fit.append(res)

            if cfg.evaluation.adapt.name == "uda":
                if isinstance(net.module_, DomainAdaptFineTuneableModel):
                    dsprep.train() # mask target domain labels
                    for du in dsprep.domains.unique():
                            domain_data = dsprep[DomainIndex(du.item())]
                            net.module_.domainadapt_finetune(x=domain_data[0]['x'], y=domain_data[1], d=domain_data[0]['d'], target_domains=target_domains)
            elif cfg.evaluation.adapt.name == "no":
                if isinstance(net.module_, FineTuneableModel):
                    dsprep.train() # mask target domain labels
                    net.module_.finetune(x=dsprep.features[train], y = dsprep.labels[train], d=dsprep.domains[train])

            duration = time() - t_start

            # save the final model
            for test_group in test_group_list:
                mdl_path = os.path.join(res_dir,'models',f'{test_group["subject"]}',f'{test_group["session"]}',f'{cfg.nnet.name}.pth')
                if not os.path.exists(os.path.split(mdl_path)[0]):
                    os.makedirs(os.path.split(mdl_path)[0])
                net.save_params(f_params=mdl_path)

            ## evaluation
            dsprep.eval() # unmask target domain labels
            
            y_hat = np.empty(dsprep.labels.shape)
            # find out latent space dimensionality
            _, l0 = net.forward(dsprep[DomainIndex(dsprep.domains[0])][0])
            l = np.empty((len(dsprep), ) + l0.shape[1:])
            
            for du in dsprep.domains.unique():
                ixs = np.flatnonzero(dsprep.domains == du)
                domain_data = dsprep[DomainIndex(du)]

                y_hat_domain, l_domain, *_ = net.forward(domain_data[0])
                y_hat_domain, l_domain = y_hat_domain.numpy().argmax(axis=1), l_domain.to(device=cpu).numpy()
                y_hat[ixs] = y_hat_domain
                l[ixs] = l_domain

            score_trn = scorefun(dsprep.labels[train_source_doms], y_hat[train_source_doms])         

            for test_group in test_group_list:
                score_tst = scorefun(dsprep.labels[test_group["idxs"]], y_hat[test_group["idxs"]])

                res = pd.DataFrame({'dataset': dataset.code, 
                                'subject': test_group["subject"], 
                                'session': test_group["session"],
                                'method': cfg.nnet.name,
                                'score_trn': score_trn,
                                'score_tst': score_tst,
                                'time': duration,
                                'n_test': len(test),
                                'classes':n_classes}, index=[resix])
                results = results.append(res)
                resix += 1
                r = res.iloc[0,:]
                log.info(f'{r.dataset} {r.classes}cl | {r.subject} | {r.session} | {r.method} :       trn={r.score_trn:.2f} tst={r.score_tst:.2f}')

            ## fine tuning
            if ft_pipeline is not None:
                # fitting
                dsprep.train() # mask target domain labels
                ft_pipeline.fit(l[train], dsprep.labels[train])
                y_hat_ft = ft_pipeline.predict(l)

                # evaluation
                dsprep.eval() # unmask target domain labels
                ft_score_trn = scorefun(dsprep.labels[train_source_doms], y_hat_ft[train_source_doms])

                for test_group in test_group_list:
                    ft_score_tst = scorefun(dsprep.labels[test_group["idxs"]], y_hat_ft[test_group["idxs"]])

                    res = pd.DataFrame({'dataset': dataset.code, 
                                        'subject': test_group["subject"], 
                                        'session':test_group["session"],
                                        'method': f'{cfg.nnet.name}+FT',
                                        'score_trn': ft_score_trn,
                                        'score_tst': ft_score_tst,
                                        'time': duration,
                                        'n_test': len(test),
                                        'classes':n_classes}, index=[resix])
                    results = results.append(res)
                    resix += 1
                    r = res.iloc[0,:]
                    log.info(f'{r.dataset} {r.classes}cl | {r.subject} | {r.session} | {r.method} :    trn={r.score_trn:.2f} tst={r.score_tst:.2f}')

    if len(results_fit):
        results_fit = pd.concat(results_fit)

        results_fit['preprocessing'] = prep_name
        results_fit['evaluation'] = cfg.evaluation.strategy
        results_fit['adaptation'] = cfg.evaluation.adapt.name

        for method in results_fit['method'].unique():
            method_res = results[results['method'] == method]
            results_fit.to_csv(os.path.join(res_dir, f'nnfitscores_{method}.csv'), index=False)

    if len(results) > 0:

        results['preprocessing'] = prep_name
        results['evaluation'] = cfg.evaluation.strategy
        results['adaptation'] = cfg.evaluation.adapt.name

        for method in results['method'].unique():
            method_res = results[results['method'] == method]
            method_res.to_csv(os.path.join(res_dir, f'scores_{method}.csv'), index=False)

        print(results.groupby('method').agg(['mean','std']))

if __name__ == '__main__':
    main()
