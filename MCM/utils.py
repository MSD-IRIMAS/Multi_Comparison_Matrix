import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
from scipy.stats import wilcoxon
from baycomp import SignedRankTest

def get_keys_for_two_comparates(a, b):
    return f"{a}-vs-{b}"

def decode_results_data_frame(df, analysis):
    """
    
    Decode the necessary information from the data frame and put them into the json analysis file.

    Parameters
    ----------

    df : pandas DataFrame containing the statistics of each comparate on multiple datasets.
         shape = (n_datasets, n_comparates), columns = [list of comparates names]
    analysis : python dictionary

    """

    df_columns = list(df.columns) # extract columns from data frame

    # check if dataset column name is correct

    if analysis['dataset-column'] is not None:
        if analysis['dataset-column'] not in df_columns:
            raise KeyError("The column "+analysis['dataset-column']+" is missing.")

    # get number of examples (datasets)
    # n_datasets = len(np.unique(np.asarray(df[analysis['dataset-column']])))
    n_datasets = len(df.index)

    analysis['n-datasets'] = n_datasets # add number of examples to dictionary

    if analysis['dataset-column'] is not None:
        
        analysis['dataset-names'] = list(df[analysis['dataset-column']]) # add example names to dict
        df_columns.remove(analysis['dataset-column']) # drop the dataset column name from columns list
        # and keep comparate names

    comparate_names = df_columns.copy()
    n_comparates = len(comparate_names)

    # add the information about comparates to dict
    analysis['comparate-names'] = comparate_names
    analysis['n-comparates'] = n_comparates

def get_pairwise_content(
        x, y,
        order_WinTieLoss='higher',
        includeProbaWinTieLoss=False,
        include_pvalue=True,
        pvalue_test='wilcoxon',
        pvalue_threshhold=0.05,
        use_mean='mean-difference',
        bayesian_rope=0.01):

    """
    
    Get the pairwise comparison between two comparates on all the given datasets.

    Parameters:
    -----------

    x : ndarray of shape = (n_datasets,) containing the statistics of a comparate x
        on all the datasets
    y : ndarray of shape = (n_datasets,) containing the statistics of a comparate y
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
    used_mean : str, default = 'mean-difference', the mean used to comapre two comparates.
    bayesian_rope : float, default = 0.01, the rope used in case include_ProbaWinTieLoss is True

    Returns
    -------
    content : python dictionary

    """

    content = {}

    if order_WinTieLoss == 'lower':

        win = len(x[x < y])
        loss = len(x[x > y])
        tie = len(x[x == y])
    
    else: # by default we assume higher is better

        win = len(x[x > y])
        loss = len(x[x < y])
        tie = len(x[x == y])

    content['win'] = win
    content['tie'] = tie
    content['loss'] = loss

    if include_pvalue:

        if pvalue_test == "wilcoxon":

            pvalue = wilcoxon(x=x, y=y, zero_method='pratt')[1]
            content['pvalue'] = pvalue

            if pvalue_test == 'wilcoxon':

                if pvalue < pvalue_threshhold:
                    content['is-significant'] = True
                else:
                    content['is-significant'] = False
            
            else:
                print(f"{pvalue_test} test is not supported yet")
    
    if includeProbaWinTieLoss:

        bayesian_test = SignedRankTest(x=x, y=y, rope=bayesian_rope)

        p_x_wins, p_rope, p_y_wins = bayesian_test.probs()

        content['p-x-wins'] = p_x_wins
        content['p-y-wins'] = p_y_wins
        content['p-rope'] = p_rope

    if use_mean == 'mean-difference':

        content['mean'] = np.mean(x) - np.mean(y)

    return content

def re_order_comparates(df_results, analysis):

    """
    
    Re order comparates given a specific order stats.

    Parameters
    ----------

    df_results : pandas DataFrame containing the results of each comparate for all datasets
    analysis : python dictionary containing the information of the pairwise comparison,
               the ordering information will be added to this dictionary
    
    """

    stats = []

    if analysis['order-stats'] == "average-statistic":

        for i in range(analysis['n-comparates']):

            stats.append(analysis['average-statistic'][analysis['comparate-names'][i]])
    
    elif analysis['order-stats'] == 'average-rank':

        if analysis['dataset-column'] is not None:
            np_results = np.asarray(df_results.drop([analysis['dataset-column']],axis=1))
        else:
            np_results = np.asarray(df_results)

        df = pd.DataFrame(columns=['comparate-name','values'])
        
        for i, comparate_name in enumerate(analysis['comparate-names']):

            for j in range(analysis['n-datasets']):

                df = df.append({'comparate-name' : comparate_name,
                                'values' : np_results[j][i]}, ignore_index=True)

        rank_values = np.array(df['values']).reshape(analysis['n-comparates'], analysis['n-datasets'])
        df_ranks = pd.DataFrame(data=rank_values)

        average_ranks = df_ranks.rank(ascending=False).mean(axis=1)

        stats = np.asarray(average_ranks)
    
    elif analysis['order-stats'] == 'max-wins':

        for i in range(analysis['n-comparates']):

            wins = []

            for j in range(analysis['n-comparates']):

                if i != j:                
                    wins.append(analysis[analysis['comparate-names'][i]+'-vs-'+analysis['comparate-names'][j]]['win'])
            
            stats.append(int(np.max(wins)))
    
    elif analysis['order-stats'] == 'amean-amean':

        for i in range(analysis['n-comparates']):

            ameans = []

            for j in range(analysis['n-comparates']):

                if i != j:
                    ameans.append(analysis[analysis['comparate-names'][i]+'-vs-'+analysis['comparate-names'][j]]['mean'])
            
            stats.append(np.mean(ameans))
    
    elif analysis['order-stats'] == 'pvalue':

        for i in range(analysis['n-comparates']):

            pvalues = []
            
            for j in range(analysis['n-comparates']):

                if i != j:
                    pvalues.append(analysis[analysis['comparate-names'][i]+'-vs-'+analysis['comparate-names'][j]]['pvalue'])
        
            stats.append(np.mean(pvalues))

    if analysis['order-better'] == 'increasing':
        ordered_indices = np.argsort(stats)
    else: # decreasing
        ordered_indices = np.argsort(stats)[::-1]

    analysis['ordered-stats'] = list(np.asarray(stats)[ordered_indices])
    analysis['ordered-comparate-names'] = list(np.asarray(analysis['comparate-names'])[ordered_indices])

def get_ticks(analysis, row_comparates, col_comparates, precision=4):

    """
    
    Generating tick labels for the heatmap.

    Parameters
    ----------

    analysis : python dictionary containing all the information about the comparates and comparisons

    Returns
    -------
    
    xticks : list of str, containing the tick labels for each classifer
    yticks : list of only one str, containing one tick of the comparate in question (proposed_method)

    """

    fmt = f"{precision}f"
    xticks = []
    yticks = []

    n_rows = len(row_comparates)
    n_cols = len(col_comparates)

    all_comparates = analysis['ordered-comparate-names']
    all_stats = analysis['ordered-stats']

    for i in range(n_rows):

        stat = all_stats[[x for x in range(len(all_comparates)) if all_comparates[x] == row_comparates[i]][0]]
        
        tick_label = f"{row_comparates[i]}\n{stat:.{fmt}}"
        yticks.append(tick_label)

    for i in range(n_cols):
        stat = all_stats[[x for x in range(len(all_comparates)) if all_comparates[x] == col_comparates[i]][0]]
        tick_label = f"{col_comparates[i]}\n{stat:.{fmt}}"
        xticks.append(tick_label)

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

def get_limits(pairwise_matrix, can_be_negative=False, precision=4):

    """
    
    Get the limits, min and max, of the heatmap color bar values
    min and max produced are equal in absolute value, to insure symmetry

    Parameters
    ----------
    pairwise_matrix : ndarray, shape = (n_comparates, n_comparates), a matrix containing 
                      the 1v1 statistical values (by default: difference of arithmetic mean of stats)
    can_be_negative : bool, default = False, whether or not the values can be negative to help
                        the case of the heatline
    
    Returns
    -------
    
    min_value : float, the min value
    max_value : float, the max value
    
    """

    if pairwise_matrix.shape[0] == 1:

        min_value = round(np.min(pairwise_matrix), precision)
        max_value = round(np.max(pairwise_matrix), precision)

        if min_value >= 0 and max_value >= 0 and (not can_be_negative):
            return min_value, max_value

        return - max(abs(min_value),abs(max_value)), max(abs(min_value),abs(max_value))

    min_value = np.min(pairwise_matrix)
    max_value = np.max(pairwise_matrix)

    if min_value < 0 or max_value < 0:
        max_min_max = max(abs(min_value), abs(max_value))

        min_value = get_sign(x=min_value) * max_min_max
        max_value = get_sign(x=max_value) * max_min_max

    return round(min_value, precision), round(max_value, precision)

def get_fig_size(
        fig_size,
        n_rows,
        n_cols,
        n_info_per_cell=None,
        longest_string=None,
):
    
    """
    
    Generate figure size given the input parameters
    
    """

    if isinstance(fig_size, str):

        if fig_size == 'auto':

            if (n_rows == 1) and (n_cols == 2):
                size = [int(max(longest_string * 0.13, 1) * n_cols), int(max(n_info_per_cell * 0.1, 1) * (n_rows + 1))]
            
            elif n_rows <= n_cols:
                size = [int(max(longest_string * 0.125, 1) * n_cols), int(max(n_info_per_cell * 0.1, 1) * (n_rows + 1))]
            
            else:
                size = [int(max(longest_string * 0.1, 1) * (n_cols + 1)), int(max(n_info_per_cell * 0.125, 1) * n_rows)]
            
            if n_rows == n_cols == 1:
                size[0] = size[0] + int(longest_string * 0.125)

            return size
        
        return [int(s) for s in fig_size.split(',')]
    
    return fig_size

def plot_1v1(x,
             y,
             name_x,
             name_y,
             win_x,
             loss_x,
             tie,
             min_lim: int = 0,
             max_lim: int = 1,
             output_directory='./',
             scatter_size: int = 100,
             linewidth: int = 3,
             linecolor: str = "black",
             fontsize: int = 20):
    
    """
    
    Plot the 1v1 scatter values of two comparates x and y on all the datasets
    Inlude the win tie loss count and the pvalue stats

    Parameters:
    -----------

    x : ndarray, shape = (n_datasets,), containing statistics of a comparate x on all datasets
    y : ndarray, shape = (n_datasets,), containing statistics of a comparate y on all datasets
    name_x : str, name of comparate x
    name_y : str, name of comparate y
    win_x : number of wins for x
    tie : number of ties
    loss_x : number of loss for y
    output_directory : str, default = './', output directory of the 1v1 plots

    Plot
    ----
    On the y-axis, the comparate y values will be scattered, and the x values on the x-axis
    The points in blue are wins for y, points in orange are wins for x
    
    """

    save_path = os.path.join(output_directory, "1v1_plots", get_keys_for_two_comparates(name_x, name_y) + ".pdf")
    if os.path.exists(save_path):
        return

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(x)):
        if x[i] > y[i]:
            ax.scatter(y[i], x[i], color='blue', s=scatter_size)
        elif x[i] < y[i]:
            ax.scatter(y[i], x[i], color='orange', s=scatter_size)
        else:
            ax.scatter(y[i], x[i], color='green', s=scatter_size)

    ax.plot([min_lim, max_lim], [min_lim, max_lim], lw=linewidth, color=linecolor)
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(name_y, fontsize=fontsize)
    ax.set_ylabel(name_x, fontsize=fontsize)

    p_value = round(wilcoxon(x=x, y=y, zero_method="pratt")[1], 4)
    legend_elements = [
        mpl.lines.Line2D([], [], marker='o', color='blue', label=f"Win {win_x}", linestyle='None'),
        mpl.lines.Line2D([], [], marker='o', color='green', label=f"Tie {tie}", linestyle='None'),
        mpl.lines.Line2D([], [], marker='o', color='orange', label=f"Loss {loss_x}", linestyle='None'),
        mpl.lines.Line2D([], [], marker=' ', color='none', label=f"P-Value {p_value}")
    ]

    ax.legend(handles=legend_elements)

    if not os.path.exists(output_directory + '1v1_plots/'):
        os.mkdir(output_directory + '1v1_plots/')
    plt.savefig(save_path, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.cla()
    plt.clf()
    plt.close()

def capitalize_label(s):

    if len(s.split('-')) == 1:
        return s.capitalize()
    
    else:
        return '-'.join(ss.capitalize() for ss in s.split('-'))

def holms_correction(analysis):

    pvalues = []

    for i in range(analysis['n-comparates']):

        comparate_i = analysis['comparate-names'][i]

        for j in range(i+1, analysis['n-comparates']):

            if i != j:

                comparate_j = analysis['comparate-names'][j]
                pairwise_key = get_keys_for_two_comparates(comparate_i, comparate_j)
                pvalues.append(analysis[pairwise_key]['pvalue'])

    pvalues_sorted = np.sort(pvalues)

    k = 0
    m = len(pvalues)

    pvalue_times_used = {}

    for pvalue in pvalues:
        pvalue_times_used[pvalue] = 0

    for i in range(analysis['n-comparates']):

        comparate_i = analysis['comparate-names'][i]
        
        for j in range(i+1, analysis['n-comparates']):

            if i !=  j:

                comparate_j = analysis['comparate-names'][j]
                pairwise_key = get_keys_for_two_comparates(comparate_i, comparate_j)
                pvalue = analysis[pairwise_key]['pvalue']
                index_pvalue = np.where(pvalues_sorted == pvalue)[0]

                if len(index_pvalue) == 1:
                    index_pvalue = index_pvalue[0]
                else:
                    index_pvalue = index_pvalue[pvalue_times_used[pvalue]]
                    pvalue_times_used[pvalue] += 1

                pvalue_threshhold_corrected = analysis['pvalue-threshold'] / (m-index_pvalue)

                if pvalue < pvalue_threshhold_corrected:
                    analysis[pairwise_key]['is-significant'] = True
                else:
                    analysis[pairwise_key]['is-significant'] = False
                
                k = k + 1
    
    for i in range(analysis['n-comparates']):

        comparate_i = analysis['comparate-names'][i]
        
        for j in range(i+1,analysis['n-comparates']):
        
            comparate_j = analysis['comparate-names'][j]

            pairwise_key_ij = get_keys_for_two_comparates(comparate_i, comparate_j)
            pairwise_key_ji = get_keys_for_two_comparates(comparate_j, comparate_i)   

            analysis[pairwise_key_ji]['is-significant'] = analysis[pairwise_key_ij]['is-significant']
            analysis[pairwise_key_ji]['pvalue'] = analysis[pairwise_key_ij]['pvalue']

def get_cell_legend(
        analysis,
        win_label='r>c',
        tie_label='r=c',
        loss_label='r<c',
):

    cell_legend = capitalize_label(analysis['use-mean'])
    longest_string = len(cell_legend)

    win_tie_loss_string = f"{win_label} / {tie_label} / {loss_label}"
    longest_string = max(longest_string, len(win_tie_loss_string))

    cell_legend = f"{cell_legend}\n{win_tie_loss_string}"

    if analysis['include-pvalue']:

        longest_string = max(longest_string, len(capitalize_label(analysis['pvalue-test'])))
        pvalue_test = capitalize_label(analysis['pvalue-test']) + ' p-value'
        cell_legend = f'{cell_legend}\n{pvalue_test}'
    
    return cell_legend, longest_string

def get_annotation(
        analysis,
        row_comparates,
        col_comparates,
        cell_legend,
        p_value_text,
        colormap='coolwarm',
        colorbar_value=None,
        precision=4
):

    fmt = f".{precision}f"

    n_rows = len(row_comparates)
    n_cols = len(col_comparates)

    pairwise_matrix = np.zeros(shape=(n_rows, n_cols))

    df_annotations = []

    n_info_per_cell = 0
    longest_string = 0

    p_value_cell_location = None
    legend_cell_location = None

    for i in range(n_rows):

        row_comparate = row_comparates[i]
        dict_to_add = {
            'comparates' : row_comparate
        }
        longest_string = max(longest_string, len(row_comparate))

        for j in range(n_cols):

            col_comparate = col_comparates[j]

            if row_comparate != col_comparate:

                longest_string = max(longest_string, len(col_comparate))
                pairwise_key = get_keys_for_two_comparates(row_comparate, col_comparate)

                if colormap is not None:
                    try:
                        pairwise_matrix[i, j] = analysis[pairwise_key][colorbar_value]
                    except:
                        pairwise_matrix[i, j] = analysis[pairwise_key]['mean']
                
                else:
                    pairwise_matrix[i, j] = 0
                
                pairwise_content = analysis[pairwise_key]
                pairwise_keys = list(pairwise_content.keys())

                string_in_cell = f"{pairwise_content['mean']:{fmt}}\n"
                n_info_per_cell = 1

                if 'win' in pairwise_keys:
                    
                    string_in_cell = f"{string_in_cell}{pairwise_content['win']} / "
                    string_in_cell = f"{string_in_cell}{pairwise_content['tie']} / "
                    string_in_cell = f"{string_in_cell}{pairwise_content['loss']}\n"
                    
                    n_info_per_cell += 1
                
                if 'p-x-wins' in pairwise_keys:

                    string_in_cell = f"{string_in_cell}{pairwise_content['p-x-wins']:{fmt}} / "
                    string_in_cell = f"{string_in_cell}{pairwise_content['p-rope']:{fmt}} / "
                    string_in_cell = f"{string_in_cell}{pairwise_content['p-y-wins']:{fmt}}\n"
                
                if 'pvalue' in pairwise_keys:

                    _p_value = round(pairwise_content['pvalue'], precision)
                    alpha = 10 ** (-precision)

                    if _p_value < alpha:
                        string_in_cell = f"{string_in_cell} $\leq$ {alpha:.0e}"
                    else:
                        string_in_cell = f"{string_in_cell}{pairwise_content['pvalue']:{fmt}}"
                    
                    n_info_per_cell += 1
                
                dict_to_add[col_comparate] = string_in_cell
            
            else:

                if legend_cell_location is None:
                    
                    dict_to_add[row_comparate] = cell_legend
                    legend_cell_location = (i, j)
                else:

                    dict_to_add[col_comparate] = '-'
                    p_value_cell_location = (i, j)
                
                pairwise_matrix[i, j] = 0.0
        
        df_annotations.append(dict_to_add)
    
    if p_value_cell_location is not None:

        col_comparate = col_comparates[p_value_cell_location[1]]
        df_annotations[p_value_cell_location[0]][col_comparate] = p_value_text
    
    df_annotations = pd.DataFrame(df_annotations)

    out = dict(
        df_annotations=df_annotations,
        pairwise_matrix=pairwise_matrix,
        n_info_per_cell=n_info_per_cell,
        longest_string=longest_string,
        legend_cell_location=legend_cell_location,
        p_value_cell_location=p_value_cell_location,
    )

    return out