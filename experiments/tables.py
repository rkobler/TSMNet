#%%
import os
import pandas as pd
from mne.stats import permutation_t_test

from resultio import *

#%% grand average table

evaluations = ['inter-session', 'inter-subject']
preprocessing = ['bb4-36Hz','fb4-36Hz']

datasets = [
    'BNCI2014001', 
    'BNCI2015001', 
    'Lee2019', 
    'Stieger2021', 
    'Lehner2021', 
    'Hehenberger2021',
    'Hinss2021'
]

methods = [
    'FBCSP+SVM',
    'TSM+SVM',
    'FB+TSM+LR',
    'EEGNet',
    'ShConvNet',
    'FBCSP+DSS+LDA',
    'URPA+MDM',
    'SPDOT+TSM+SVM',
    'EEGNet+DANN',
    'ShConvNet+DANN',
    'TSMNet+SPDDSMBN',
]
reference_method = 'TSMNet+SPDDSMBN'


resdf = loadresults(datasets, evaluations, preprocessing, methods)

# average results across subjects/sessions for each dataset
summarydf = summarize_within_evaluation(resdf.set_index(['dataset', 'evaluation', 'method', 'subject', 'session']), 
    {"score_tst": ['mean', 'std']}, 
    ['dataset', 'evaluation', 'adaptation','method']).dropna(axis=0,how='all')


# compute difference to reference method and summarize across datasets
diffdf = difference_within_evaluation(resdf.set_index(['dataset', 'evaluation', 'method', 'subject', 'session']), reference_method, columns= ['score_trn', 'score_tst'])

# compute statistical tests
testdf = diffdf.set_index('adaptation', append=True).dropna(how='all')['score_tst'].\
    unstack(['method', 'adaptation']).drop(columns=reference_method)

def grouplevel_tests(grp):
    grp = grp.dropna(1,'all').dropna(0,'any')
    t,p, _ = permutation_t_test(grp.to_numpy(), 
                                n_permutations=1e4, tail=0, n_jobs=1, seed=None, verbose=None)
    statdf = pd.DataFrame({'tval': t, 'pval': p}, index=grp.columns)
    statdf['significant'] = statdf['pval'] <= 0.05
    statdf['ntest'] = grp.shape[1]  
    statdf['df'] = grp.shape[0] - 1 
    return statdf

statdf = testdf.groupby(['dataset', 'evaluation']).apply(grouplevel_tests)

statdf['metric'] = 'score_tst'
statdf = statdf.set_index('metric', append=True).unstack('metric').swaplevel(0,1, 1)
statdf.columns = statdf.columns.set_names([None,None])
statdf.index = statdf.index.swaplevel(-1,-2)

# combine results and format dataframe as latex table
tabledf = summarydf.join(statdf)

os.makedirs('tables', exist_ok=True)
tabledf.to_csv('tables/tab-score_tst-methods.csv')

def summarize_score(x):
    decorator = ' '
    if x['pval'] <= 0.05:
        decorator = '\\sig'
    if x['pval'] <= 0.01:
        decorator = '\\ssig'
    if x['pval'] <= 0.001:
        decorator = '\\sssig'

    return f"{decorator}{x['mean']*100:4.1f} ({x['std']*100:4.1f})".replace(' ', '~')

# create summarizing tables
tabledf.loc[:,('Δ balanced accuracy (%)', 'mean (std)')] = tabledf['score_tst'].apply(
    summarize_score,
     axis=1)

tabledf = tabledf.unstack(['dataset', 'evaluation']).dropna(axis=1,how='all').\
    drop(columns=['score_tst']).droplevel([0,1], axis=1)

os.makedirs('tables', exist_ok=True)
with open('tables/tab-score_tst-methods.tex', 'wt') as f:
    f.write(tabledf[tabledf.columns.get_level_values(0).categories[:2]].to_latex(escape=False).replace('~','\\phantom{0}'))
    f.write(tabledf[tabledf.columns.get_level_values(0).categories[2:4]].to_latex(escape=False).replace('~','\\phantom{0}'))
    f.write(tabledf[tabledf.columns.get_level_values(0).categories[4:7]].to_latex(escape=False).replace('~','\\phantom{0}'))

tabledf

#%% ablation study 

evaluations = ['inter-session']
preprocessing = ['bb4-36Hz']

datasets = {
    'BNCI2014001': 'BNCI2014001', 
    'BNCI2015001' : 'BNCI2015001', 
    'Lee2019' : 'Lee2019', 
    'Stieger2021' : 'Stieger2021', 
    'Lehner2021' : 'Lehner2021', 
}

methods = [
    'TSMNet+SPDDSMBN',
    'TSMNet+SPDDSBN',
    'TSMNet+SPDMBN',
    'CNNNet+DSMBN',
    'CNNNet+MBN',
]

reference_method = 'TSMNet+SPDDSMBN'

resdf = loadresults(datasets, evaluations, preprocessing, methods)


resdf = resdf.set_index(['dataset', 'evaluation', 'method', 'subject', 'session'])

diffdf = difference_within_evaluation(resdf, reference_method, 
                                      columns= ['score_trn', 'score_tst'])

# average results across sessions
summarydf = summarize_within_evaluation(diffdf, 
    {"score_tst": ['mean', 'std'], "time": ['mean', 'std']}, 
    ['dataset', 'subject', 'method']).dropna(how='all')

# compute statistical tests
testdf = summarydf.loc[:,('score_tst', 'mean')].unstack(['method']).drop(columns=reference_method)
t,p, _ = permutation_t_test(testdf.to_numpy(), 
                            n_permutations=1e4, tail=0, n_jobs=1, seed=None, verbose=None)

statdf = pd.DataFrame({'tval': t, 'pval': p}, index=testdf.columns)
statdf['significant'] = statdf['pval'] <= 0.05
statdf['ntest'] = testdf.shape[1]  
statdf['df'] = testdf.shape[0] - 1 
statdf['metric'] = 'score_tst'
statdf = statdf.set_index('metric', append=True).unstack('metric').swaplevel(0,1, 1)
statdf.columns = statdf.columns.set_names([None,None])

tabledf = summarydf.groupby(['method']).mean().dropna(how='all')
tabledf = tabledf.join(statdf.loc[:,('score_tst', ['tval','pval'])], on=['method'])

os.makedirs('tables', exist_ok=True)
tabledf.to_csv('tables/tab-score_tst-ablation.csv')

tabledf.loc[:,('Δ balanced accuracy (%)', 'mean (std)')] = tabledf['score_tst'].apply(
    lambda x : f"{x['mean']*100:4.1f} ({x['std']*100:4.1f})".replace(' ', '~'),
     axis=1)
tabledf.loc[:,('Δ balanced accuracy (%)', 't-val (p-val)')] = tabledf['score_tst'].apply(
    lambda x : f"{x['tval']:3.1f} ({x['pval']:5.4f})".replace(' ', '~'),
     axis=1)
tabledf.loc[:,('fit time (s)', '')] = tabledf['time'].apply(
    lambda x : f"{x['mean']:4.1f} ({x['std']:4.1f})".replace(' ', '~'),
     axis=1)

tabledf = tabledf.drop(columns=['score_tst', 'time'])

os.makedirs('tables', exist_ok=True)
with open('tables/tab-score_tst-ablation.tex', 'wt') as f:
    f.writelines(tabledf.to_latex(escape=False).replace('~','\\phantom{0}').replace('Δ','\\Delta'))

tabledf
