import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
from scipy.stats import wilcoxon
from baycomp import SignedRankTest

def decode_results_data_frame(df, analysis):

    df_columns = list(df.columns)

    if analysis['dataset-column'] not in df_columns:
        raise KeyError("The column "+analysis['dataset-column']+" is missing.")

    n_datasets = len(np.unique(np.asarray(df[analysis['dataset-column']])))

    analysis['n-datasets'] = n_datasets

    analysis['dataset-names'] = list(df[analysis['dataset-column']])
    
    df_columns.remove(analysis['dataset-column'])

    classifier_names = df_columns.copy()
    n_classifiers = len(classifier_names)

    analysis['classifier-names'] = classifier_names
    analysis['n-classifiers'] = n_classifiers

def get_pairwise_content(x,
                         y,
                         order_WinTieLoss='higher',
                         includeProbaWinTieLoss=True,
                         include_pvalue=True,
                         pvalue_test='wilcoxon',
                         pvalue_correction=None,
                         pvalue_threshhold=0.05,
                         used_mean='arithmetic-mean',
                         bayesian_rope=0.01):

    content = {}

    if order_WinTieLoss == 'higher':

        win = len(x[x > y])
        loss = len(x[x < y])
        tie = len(x[x == y])
    
    elif order_WinTieLoss == 'lower':

        win = len(x[x < y])
        loss = len(x[x > y])
        tie = len(x[x == y])

    content['win'] = win
    content['tie'] = tie
    content['loss'] = loss

    if include_pvalue:

        if pvalue_test == "wilcoxon":

            pvalue = wilcoxon(x=x, y=y, zero_method='pratt')[1]
            content['pvalue'] = pvalue

            if pvalue_correction is None:

                if pvalue < pvalue_threshhold:
                    content['is-significant'] = True
                else:
                    content['is-significant'] = False
    
    if includeProbaWinTieLoss:

        bayesian_test = SignedRankTest(x=x, y=y, rope=bayesian_rope)

        p_x_wins, p_rope, p_y_wins = bayesian_test.probs()

        content['p-x-wins'] = p_x_wins
        content['p-y-wins'] = p_y_wins
        content['p-rope'] = p_rope

    if used_mean == 'arithmetic-mean':

        content['mean'] = np.mean(x) - np.mean(y)

    return content

def re_order_classifiers(df_results, analysis):

    stats = []

    if analysis['order-stats'] == "average-statistic":

        for i in range(analysis['n-classifiers']):

            stats.append(analysis['average-statistic'][analysis['classifier-names'][i]])
    
    elif analysis['order-stats'] == 'average-rank':

        np_results = np.asarray(df_results.drop([analysis['dataset-column']],axis=1))
        df = pd.DataFrame(columns=['classifier-name','values'])
        
        for i, classifier_name in enumerate(analysis['classifier-names']):

            for j in range(analysis['n-datasets']):

                df = df.append({'classifier-name' : classifier_name,
                                'values' : np_results[j][i]}, ignore_index=True)

        rank_values = np.array(df['values']).reshape(analysis['n-classifiers'], analysis['n-datasets'])
        df_ranks = pd.DataFrame(data=rank_values)

        average_ranks = df_ranks.rank(ascending=False).mean(axis=1)

        stats = np.asarray(average_ranks)
    
    elif analysis['order-stats'] == 'max-wins':

        for i in range(analysis['n-classifiers']):

            wins = []

            for j in range(analysis['n-classifiers']):

                if i != j:                
                    wins.append(analysis[analysis['classifier-names'][i]+'-vs-'+analysis['classifier-names'][j]]['win'])
            
            stats.append(int(np.max(wins)))
    
    elif analysis['order-stats'] == 'amean-amean':

        for i in range(analysis['n-classifiers']):

            ameans = []

            for j in range(analysis['n-classifiers']):

                if i != j:
                    ameans.append(analysis[analysis['classifier-names'][i]+'-vs-'+analysis['classifier-names'][j]]['mean'])
            
            stats.append(np.mean(ameans))

    if analysis['order-better'] == 'increasing':
        ordered_indices = np.argsort(stats)
    elif analysis['order-better'] == 'decreasing':
        ordered_indices = np.argsort(stats)[::-1]

    analysis['ordered-stats'] = list(np.asarray(stats)[ordered_indices])
    analysis['ordered-classifier-names'] = list(np.asarray(analysis['classifier-names'])[ordered_indices])

def get_ticks(analysis, round_number):

    ticks = []

    if analysis['order-stats'] == 'average-statistic':
        ordering = 'average-'+analysis['used-statistics']
    else:
        ordering = analysis['order-stats']

    for i in range(analysis['n-classifiers']):
        ticks.append(analysis['ordered-classifier-names'][i] + '\n' + ordering + '\n' + str(round(analysis['ordered-stats'][i], round_number)))

    return ticks

def get_ticks_heatline(analysis, proposed_method, round_number):

    xticks = []
    yticks = []

    for i in range(analysis['n-classifiers']):

        if analysis['ordered-classifier-names'][i] == proposed_method:
            yticks.append(proposed_method + ' VS')
        
        xticks.append(analysis['ordered-classifier-names'][i] + '\n' + analysis['order-stats'] + '\n' + str(round(analysis['ordered-stats'][i], round_number)))
        
    return xticks, yticks

def abs(x):

    return x if x > 0 else -x

def get_sign(x):

    return 1 if x > 0 else -1

def get_limits(pairwise_matrix):

    if pairwise_matrix.shape[0] == 1:

        min_value = round(np.min(pairwise_matrix), 4)
        max_value = round(np.max(pairwise_matrix), 4)

        return - max(abs(min_value),abs(max_value)), max(abs(min_value),abs(max_value))

    min_value = 1e9
    max_value = -1e9

    for i in range(len(pairwise_matrix)):
        
        min_i = np.min(np.delete(arr=pairwise_matrix[i], obj=i))
        
        if min_i < min_value:
            min_value = min_i

        max_i = np.max(np.delete(arr=pairwise_matrix[i], obj=i))

        if max_i > max_value:
            max_value = max_i
    
    if min_value < 0 or max_value < 0:

        max_min_max = max(abs(min_value), abs(max_value))

        min_value = get_sign(x=min_value) * max_min_max
        max_value = get_sign(x=max_value) * max_min_max

    return round(min_value, 4), round(max_value, 4)

def get_fig_size(fig_size,
                 n_classifiers,
                 pixels_per_clf_hieght,
                 pixels_per_clf_width,
                 n_info_per_line=None):

    if fig_size == 'auto':

        if n_info_per_line is not None:
            pixels_per_clf_hieght = n_info_per_line / pixels_per_clf_hieght
        return [pixels_per_clf_width * (2 + n_classifiers), pixels_per_clf_hieght * (2 + n_classifiers)]

    return [int(s) for s in fig_size.split(',')]

def get_fig_size_line(fig_size, n_classifiers, pixels_per_clf_hieght, pixels_per_clf_width):

    if fig_size == 'auto':
        return [pixels_per_clf_width * (2 + n_classifiers), pixels_per_clf_hieght * (3 + n_classifiers)]

    return [int(s) for s in fig_size.split(',')]

def plot_1v1(x,
             y,
             name_x,
             name_y,
             win_x,
             loss_x,
             tie,
             output_directory='./'):

    if os.path.exists(output_directory+'1v1_plots/'+name_x+'_vs_'+name_y+'.pdf'):
        return

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(x)):

        if x[i] > y[i]:
            ax.scatter(y[i], x[i], color='blue', s=100)
        elif x[i] < y[i]:
            ax.scatter(y[i], x[i], color='orange', s=100)
        else:
            ax.scatter(y[i], x[i], color='green', s=100)
    
    ax.plot([0,1],[0,1], lw=3, color='black')

    ax.set_xlim(0.0,1.0)
    ax.set_ylim(0.0,1.0)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(name_x, fontsize=20)
    ax.set_ylabel(name_y, fontsize=20)

    legend_elements = [
        mpl.lines.Line2D([],[], marker='o', color='blue', label='Win '+str(loss_x), linestyle='None'),
        mpl.lines.Line2D([],[], marker='o', color='green', label='Tie '+str(tie), linestyle='None'),
        mpl.lines.Line2D([],[], marker='o', color='orange', label='Loss '+str(win_x), linestyle='None'),
        mpl.lines.Line2D([],[], marker=' ', color='none', label='P-Value '+str(round(wilcoxon(x=x,y=y,zero_method='pratt')[1],4)))
    ]

    ax.legend(handles=legend_elements)

    if not os.path.exists(output_directory+'1v1_plots/'):
        os.mkdir(output_directory+'1v1_plots/')
    plt.savefig(output_directory + '1v1_plots/'+name_x+'_vs_'+name_y+'.pdf')