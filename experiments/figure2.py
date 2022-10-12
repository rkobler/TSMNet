#%%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from resultio import *

#%% grand average motor imagery results

evaluations = ['inter-session', 'inter-subject']
preprocessing = ['bb4-36Hz','fb4-36Hz']

datasets = [
    'BNCI2014001', 
    'BNCI2015001', 
    'Lee2019', 
    'Stieger2021', 
    'Lehner2021', 
    # 'Hehenberger2021',
    # 'Hinss2021'
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

size_methods = ['FBCSP+DSS+LDA', 'URPA+MDM', 'EEGNet+DANN', 'TSMNet+SPDDSMBN']
size_datasets = ['BNCI2014001', 'BNCI2015001', 'Lee2019', 'Stieger2021']

#%%

resdf = loadresults(datasets, evaluations, preprocessing, methods)

summarydf = summarize_within_evaluation(resdf, 
    {"score_tst": lambda x : np.mean(x) * 100}, ['dataset', 'evaluation', 'method', 'subject'])
summarydf = summarydf.dropna(how='all')

refdf = loadresults(datasets, ['within-session'], ['fb4-36Hz'], {'FB+TSM+LR': 'reference'})
refdf['evaluation'] = 'inter-session'

refdf_subjectlevel = summarize_within_evaluation(refdf, 
    {"score_tst": 'mean', 'evaluation': 'first'}, ['dataset', 'method', 'subject']).dropna(how='all').reset_index()
refdf_subjectlevel['session'] = -1
refdf_subjectlevel['evaluation'] = 'inter-subject'
refdf = pd.concat([refdf, refdf_subjectlevel])

methodvals = resdf.method.cat.categories.copy()
resdf = pd.concat([resdf, refdf], ignore_index=True)

resdf['method'] = resdf['method'].astype('category')
resdf['method'].cat.set_categories(methodvals.append(refdf.method.cat.categories), ordered=True, inplace=True)

resdf['evaluation'] = resdf['evaluation'].astype('category')
resdf['evaluation'].cat.set_categories(evaluations, ordered=True, inplace=True)

#%%

diffdf = difference_within_evaluation(resdf.set_index(['dataset', 'evaluation', 'method', 'subject', 'session']), 
                                      'reference', columns= ['score_tst'])

summarydiffdf = summarize_within_evaluation(diffdf, 
    {"score_tst": lambda x : np.mean(x) * 100}, ['dataset', 'evaluation', 'method', 'subject'])
summarydiffdf = summarydiffdf.dropna(how='all')

summarydiffdf = summarydiffdf.join(resdf.groupby(summarydiffdf.index.names)[['adaptation', 'category']].first()).\
    reset_index()

summarydiffdf['category'] = summarydiffdf['category'].astype('category')
summarydiffdf['category'].cat.set_categories(['component','geometric','end-to-end','proposed'], ordered=True, inplace=True)

summarydiffdf = summarydiffdf[summarydiffdf.method != 'reference']
summarydiffdf['method'].cat.set_categories(methodvals, ordered=True, inplace=True)

# %% grand average results per evaluation

palette = 'tab10'

g = sns.catplot(data=summarydiffdf, x="method", y="score_tst", 
                hue="category", 
                col='evaluation', 
                sharex=True, sharey=True, n_boot=1e3, 
                kind='bar', dodge=False,
                height=3, aspect=0.8)

ylim = [-40, 5]

for key, ax in g.axes_dict.items():

    xs = []   

    for p, l in zip(ax.patches, ax.lines):
        p.set_width(0.8)
        h, w, x = p.get_height(), p.get_width(), p.get_x()
        y =  p.get_y()
        if np.isnan(h):
            continue

        color = 'black'
        if h < 0:
            if l.get_ydata().max() < -3:
                y = l.get_ydata().min() - 1
                va = 'top'
            else:
                y = l.get_ydata().min() - 1
                va = 'top'
        else:
            y = 0 -1
            va = 'top'

        text = f'{h :.0f}'
        xy = (x + w / 2., y)
        ax.annotate(text=text, xy=xy, ha='center', va=va, size='small', color=color)
        xs.append(x)

    for ix, method in enumerate(summarydiffdf.method.cat.categories):
        xy = (ix + w / 4., ylim[0] + 0.5)
        ax.annotate(text=method, xy=xy, ha='center', va='bottom', size='small', color='black',rotation=90)
    ax.axhline(0, lw=1, color='black')

g.set(ylabel='Δ balanced accuracy (%)', ylim=ylim, xticklabels=[])

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig-figure2a.pdf', transparent=True)

#%%

sizes = resdf.groupby(['dataset', 'evaluation']).\
    agg({'n_train': 'mean', 'classes': 'mean'}).\
        sort_values('n_train').dropna(how='any').reset_index()
sizes['n_train'] = sizes['n_train']
sizes['classes'] = sizes['classes'].astype(np.int)
sizes['opc'] = (sizes['n_train'] / sizes['classes']).round(-1)

sizes['opc_label'] = sizes.apply(
    lambda x : f"{x['opc']:.0f}",
     axis=1)
sizes = sizes.set_index(['dataset', 'evaluation'])

tble_size = summarydiffdf.join(
    sizes, how='outer', on=['dataset', 'evaluation'])
tble_size = tble_size.sort_values('opc')

tble_size = tble_size[tble_size.method.isin(size_methods)]
tble_size = tble_size[tble_size.dataset.isin(size_datasets)]
tble_size.method = tble_size.method.cat.remove_unused_categories()

g = sns.catplot(data=tble_size, y="score_tst", x="opc_label", hue="method", 
                col='evaluation', 
                sharex=False, sharey=True, n_boot=100, 
                kind='box',fliersize=0., dodge=True,
                height=3, aspect=0.75)
g.map_dataframe(sns.stripplot,  
                x='opc_label', y='score_tst', hue="method", 
                alpha=0.33, linewidth=0.5, palette=palette, dodge=True)

for evaluation, ax in g.axes_dict.items():

    for label in ax.get_xticklabels():
        opc_label = label.get_text()
        x = label.get_position()[0]
        txt = tble_size[(tble_size.evaluation == evaluation) & \
            (tble_size.opc_label == opc_label)].dataset.min()
        xy = (x, ax.get_ylim()[0]+1 + 5 * (x % 2))
        ax.annotate(text=txt, xy=xy, ha='center', va='bottom', size='small')

g.set(xlabel='training set size (# obs. per class)')
g.set(ylabel='Δ balanced accuracy (%)')
for key, ax in g.axes_dict.items():
    ax.axhline(0, c='k', lw=1., zorder=0)

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig-figure2b.pdf', transparent=True)
