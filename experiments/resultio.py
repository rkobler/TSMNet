#%%

import pandas as pd
import glob

def loadresults(datasets, evaluations, preprocessing, methods):

    frames = []

    def input_to_tuple(input):
        if isinstance(input, dict):
            return input.keys(), input.values()
        else:
            return input, input

    methodkeys, methodvals = input_to_tuple(methods)
    datasetkeys, datasetvals = input_to_tuple(datasets)
    evaluationkeys, evaluationvals = input_to_tuple(evaluations)

    for dataset in datasetkeys:
        for prep in preprocessing:
            for evaluation in evaluationkeys:
                for name in sorted(glob.glob(f'outputs/{dataset}/{evaluation}/{prep}/scores_*.csv')):
                    df = pd.read_csv(name)
                    m = df.method.unique()
                    if len(m) != 1:
                        print(name)
                    assert(len(m) == 1)
                    if m[0] not in methods:
                        continue
                    if df.method.str.contains("CSP").any():
                        df['category'] = 'component'
                    elif df.method.str.contains("TSMNet").any():
                        df['category'] = 'proposed'
                    elif df.method.str.contains("Net").any():
                        df['category'] = 'end-to-end'
                    else:
                        df['category'] = 'geometric'
                    # if isinstance(methods, dict):
                    #     df['method'] = methodkeys[m[0]]
                    # if isinstance(datasets, dict):
                    #     df['dataset'] = datasets[dataset]
                    # else:
                    df['dataset'] = dataset
                    df['n_train'] = 0
                    if evaluation == 'inter-subject':
                        df['n_train'] = df.n_test.sum() - df.n_test.mean()
                    else:
                        for subject in df.subject.unique():
                            n_obs = df[df.subject == subject].n_test.sum()
                            df.loc[df.subject == subject, 'n_train'] = n_obs - df.loc[df.subject == subject, 'n_test']

                    frames += [df]

    if len(frames) > 0:
        resdf = pd.concat(frames, ignore_index=True)
    else:
        resdf = pd.DataFrame([])

    resdf['category'] = resdf['category'].astype('category')
    resdf['category'].cat.set_categories(['component','geometric','end-to-end','proposed'], ordered=True, inplace=True)
    
    resdf['adaptation'] = resdf['adaptation'].astype('category')
    resdf['adaptation'].cat.set_categories(['no','uda'], ordered=True, inplace=True)
    
    resdf['method'] = resdf['method'].astype('category')
    resdf['method'].cat.set_categories(methodkeys, ordered=True, inplace=True)
    resdf['method'] = resdf['method'].cat.rename_categories(methodvals)

    resdf['dataset'] = resdf['dataset'].astype('category')
    resdf['dataset'].cat.set_categories(datasetkeys, ordered=True, inplace=True)
    resdf['dataset'] = resdf['dataset'].cat.rename_categories(datasetvals)
    # datasets = [cat.split('_')[0] for cat in resdf['dataset'].cat.categories]
    # resdf['dataset'] = resdf['dataset'].cat.rename_categories(datasets)

    resdf['evaluation'] = resdf['evaluation'].astype('category')
    resdf['evaluation'].cat.set_categories(evaluationkeys, ordered=True, inplace=True)
    resdf['evaluation'] = resdf['evaluation'].cat.rename_categories(evaluationvals)

    return resdf


def difference_within_evaluation(resdf, reference_method, columns = ['score_trn', 'score_tst']):

    diffdf = resdf.unstack('method')
    refdf = diffdf.loc[:, (columns, reference_method)]

    for method in diffdf.columns.get_level_values('method').categories:
        diffdf.loc[:, (columns, method)] = diffdf.loc[:, (columns, method)] - refdf.to_numpy()

    diffdf = diffdf.stack('method').dropna(axis=0, how='all')
    return diffdf


def summarize_within_evaluation(resdf, aggargs, grouplevels = ['dataset', 'evaluation', 'adaptation','method'], ):

    
    if 'evaluation' in resdf.columns:
        isindex = False
        evaluations = resdf['evaluation'].unique()
    else:
        isindex = True
        evaluations = resdf.index.get_level_values('evaluation').categories
    frames = []
    for evaluation in evaluations:

        if evaluation == 'within-session':
            grouplevels_ex = list(set(grouplevels + ['subject']))
        if evaluation == 'inter-session':
            grouplevels_ex = list(set(grouplevels + ['subject']))
        else:
            grouplevels_ex = grouplevels

        if isindex:
            mask = resdf.index.get_level_values("evaluation") == evaluation
        else:
            mask = resdf['evaluation'] == evaluation

        frames += [resdf[mask].\
            groupby(grouplevels_ex).\
                agg(aggargs).\
                    groupby(grouplevels).\
                        mean()]
    
    return pd.concat(frames)

# %%
