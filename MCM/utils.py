import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
from scipy.stats import wilcoxon
from baycomp import SignedRankTest

stats_mapping = {

    "mean-"

}

def decode_results_data_frame(df, analysis):
    """
    
    Decode the necessary information from the data frame and put them into the json analysis file.

    Parameters
    ----------

    df : pandas DataFrame containing the statistics of each classifier on multiple datasets.
         shape = (n_datasets, n_classifiers), columns = [list of classifiers names]
    analysis : python dictionary

    """

    df_columns = list(df.columns) # extract columns from data frame

    # check if dataset column name is correct
    if analysis['dataset-column'] not in df_columns:
        raise KeyError("The column "+analysis['dataset-column']+" is missing.")

    # get number of examples (datasets)
    n_datasets = len(np.unique(np.asarray(df[analysis['dataset-column']])))

    analysis['n-datasets'] = n_datasets # add number of examples to dictionary

    analysis['dataset-names'] = list(df[analysis['dataset-column']]) # add example names to dict
    
    df_columns.remove(analysis['dataset-column']) # drop the dataset column name from columns list
    # and keep classifier names

    classifier_names = df_columns.copy()
    n_classifiers = len(classifier_names)

    # add the information about classifiers to dict
    analysis['classifier-names'] = classifier_names
    analysis['n-classifiers'] = n_classifiers

def get_pairwise_content(x,
                         y,
                         order_WinTieLoss='higher',
                         includeProbaWinTieLoss=False,
                         include_pvalue=True,
                         pvalue_test='wilcoxon',
                         pvalue_correction=None,
                         pvalue_threshhold=0.05,
                         used_mean='mean-difference',
                         bayesian_rope=0.01):

    """
    
    Get the pairwise comparison between two classifiers on all the given datasets.

    Parameters:
    -----------

    x : ndarray of shape = (n_datasets,) containing the statistics of a classifier x
        on all the datasets
    y : ndarray of shape = (n_datasets,) containing the statistics of a classifier y
        on all the datasets
    
    order_WinTieLoss : str, default = 'higher', the order on considering a win or a loss
                       for a given statistics
    includeProbaWinTieLoss : bool, default = False, condition whether or not include
                             the bayesian test of [1] for a probabilistic win tie loss count
    include_pvalue : bool, default = True, condition whether or not include a pvalue stats
    pvalue_test : str, default = 'wilcoxon', the statistical test to produce the pvalue stats.
    pvalue_correction : str, default = None, which correction to use for the pvalue significant test
    pvalue_threshhold : float, default = 0.05, threshold for considering a comparison is significant
                        or not. If pvalue < pvalue_threshhold -> comparison is significant.
    used_mean : str, default = 'mean-difference', the mean used to comapre two classifiers.
    bayesian_rope : float, default = 0.01, the rope used in case include_ProbaWinTieLoss is True

    Returns
    -------
    content : python dictionary

    """

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

    if used_mean == 'mean-difference':

        content['mean'] = np.mean(x) - np.mean(y)

    return content

def re_order_classifiers(df_results, analysis):

    """
    
    Re order classifiers given a specific order stats.

    Parameters
    ----------

    df_results : pandas DataFrame containing the results of each classifier for all datasets
    analysis : python dictionary containing the information of the pairwise comparison,
               the ordering information will be added to this dictionary
    
    """

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
    
    elif analysis['order-stats'] == 'pvalue':

        for i in range(analysis['n-classifiers']):

            pvalues = []
            
            for j in range(analysis['n-classifiers']):

                if i != j:
                    pvalues.append(analysis[analysis['classifier-names'][i]+'-vs-'+analysis['classifier-names'][j]]['pvalue'])
        
            stats.append(np.mean(pvalues))

    if analysis['order-better'] == 'increasing':
        ordered_indices = np.argsort(stats)
    elif analysis['order-better'] == 'decreasing':
        ordered_indices = np.argsort(stats)[::-1]

    analysis['ordered-stats'] = list(np.asarray(stats)[ordered_indices])
    analysis['ordered-classifier-names'] = list(np.asarray(analysis['classifier-names'])[ordered_indices])

def get_ticks(analysis):

    """
    
    Generating tick labels for the heatmap.

    Parameters
    ----------

    analysis : python dictionary containing all the information about the classifiers and comparisons

    Returns
    -------
    
    xticks : list of str, containing the tick labels for each classifer
    yticks : list of only one str, containing one tick of the classifier in question (proposed_method)

    """

    xticks = []
    yticks = []

    if analysis['order-stats'] == 'average-statistic':
        ordering = 'average-'+analysis['used-statistics']
    else:
        ordering = analysis['order-stats']

    for i in range(analysis['n-classifiers']):
        yticks.append(analysis['ordered-classifier-names'][i])
        xticks.append(analysis['ordered-classifier-names'][i] + '\n' + ordering + '\n' + str(round(analysis['ordered-stats'][i], 4)))

    return xticks, yticks

def get_ticks_heatline(analysis, proposed_method):

    """
    
    Generating tick labels for the heat line.

    Parameters
    ----------

    analysis : python dictionary containing all the information about the classifiers and comparisons
    proposed_method : str, name of the classifier that you want to compare with all the others
    
    Returns
    -------

    xticks : list of str, containing the tick labels for each classifer
    yticks : list of only one str, containing one tick of the classifier in question (proposed_method)
    
    """

    xticks = []
    yticks = []

    if analysis['order-stats'] == 'average-statistic':
        ordering = 'average-'+analysis['used-statistics']
    else:
        ordering = analysis['order-stats']

    for i in range(analysis['n-classifiers']):

        if analysis['ordered-classifier-names'][i] == proposed_method:
            yticks.append(proposed_method + ' VS')
        
        xticks.append(analysis['ordered-classifier-names'][i] + '\n' + ordering + '\n' + str(round(analysis['ordered-stats'][i], 4)))
        
    return xticks, yticks

def abs(x):

    """
    
    Absolute value.
    
    """

    return x if x > 0 else -x

def get_sign(x):

    """
    
    Sign of a value

    """

    return 1 if x > 0 else -1

def get_limits(pairwise_matrix, can_be_negative=False):

    """
    
    Get the limits, min and max, of the heatmap color bar values
    min and max produced are equal in absolute value, to insure symmetry

    Parameters
    ----------
    pairwise_matrix : ndarray, shape = (n_classifiers, n_classifiers), a matrix containing 
                      the 1v1 statistical values (by default: difference of arithmetic mean of stats)
    can_be_negative : bool, default = False, whether or not the values can be negative to help
                        the case of the heatline
    
    Returns
    -------
    
    min_value : float, the min value
    max_value : float, the max value
    
    """

    if pairwise_matrix.shape[0] == 1:

        min_value = round(np.min(pairwise_matrix), 4)
        max_value = round(np.max(pairwise_matrix), 4)

        if min_value >= 0 and max_value >= 0 and (not can_be_negative):
            return min_value, max_value

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
    
    """
    
    Generate figure size given the input parameters
    
    """

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
    
    """
    
    Plot the 1v1 scatter values of two classifiers x and y on all the datasets
    Inlude the win tie loss count and the pvalue stats

    Parameters:
    -----------

    x : ndarray, shape = (n_datasets,), containing statistics of a classifier x on all datasets
    y : ndarray, shape = (n_datasets,), containing statistics of a classifier y on all datasets
    name_x : str, name of classifier x
    name_y : str, name of classifier y
    win_x : number of wins for x
    tie : number of ties
    loss_x : number of loss for y
    output_directory : str, default = './', output directory of the 1v1 plots

    Plot
    ----
    On the y-axis, the classifier y values will be scattered, and the x values on the x-axis
    The points in blue are wins for y, points in orange are wins for x
    
    """

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

    ax.set_xlabel(name_y, fontsize=20)
    ax.set_ylabel(name_x, fontsize=20)

    legend_elements = [
        mpl.lines.Line2D([],[], marker='o', color='blue', label='Win '+str(win_x), linestyle='None'),
        mpl.lines.Line2D([],[], marker='o', color='green', label='Tie '+str(tie), linestyle='None'),
        mpl.lines.Line2D([],[], marker='o', color='orange', label='Loss '+str(loss_x), linestyle='None'),
        mpl.lines.Line2D([],[], marker=' ', color='none', label='P-Value '+str(round(wilcoxon(x=x,y=y,zero_method='pratt')[1],4)))
    ]

    ax.legend(handles=legend_elements)

    if not os.path.exists(output_directory+'1v1_plots/'):
        os.mkdir(output_directory+'1v1_plots/')
    plt.savefig(output_directory + '1v1_plots/'+name_x+'_vs_'+name_y+'.pdf')