import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from .utils import *

class NpEncoder(json.JSONEncoder):

    """
    
    Encoder for json files saving.
    
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def get_analysis(df_results_path,
                 output_dir='./',
                 used_statistic='Accuracy',
                 save_as_json=True,
                 plot_1v1_comparisons=False,
                 order_WinTieLoss='higher',
                 include_ProbaWinTieLoss=False,
                 bayesian_rope=0.01,
                 include_pvalue=True,
                 pvalue_test='wilcoxon',
                 pvalue_correction=None,
                 pvalue_threshhold=0.05,
                 used_mean='arithmetic-mean',
                 order_stats='average-statistic',
                 order_better='decreasing',
                 dataset_column='dataset_name',):
    
    """
    
    Get analysis of all the pairwise and multi classifier comparisons and store them in analysis
    python dictionary. With a boolean parameter, you can plot the 1v1 scatter results.

    Parameters
    ----------

    df_results_path : str, the path to the csv file containing results
    output_dir : str, default = './', the output directory for the results
    used_statistic : str, default = 'Accuracy', one can imagine using error, time, memory etc. instead
    save_as_json : bool, default = True, whether or not to save the python analysis dict
                   into a json file format
    plot_1v1_comparisons : bool, default = True, whether or not to plot the 1v1 scatter results


    order_WinTieLoss : str, default = 'higher', the order on considering a win or a loss
                       for a given statistics
    includeProbaWinTieLoss : bool, default = False, condition whether or not include
                             the bayesian test of [1] for a probabilistic win tie loss count
    include_pvalue : bool, default = True, condition whether or not include a pvalue stats
    pvalue_test : str, default = 'wilcoxon', the statistical test to produce the pvalue stats.
    pvalue_correction : str, default = None, which correction to use for the pvalue significant test
    pvalue_threshhold : float, default = 0.05, threshold for considering a comparison is significant
                        or not. If pvalue < pvalue_threshhold -> comparison is significant.
    used_mean : str, default = 'arithmetic', the mean used to comapre two classifiers.
    bayesian_rope : float, default = 0.01, the rope used in case include_ProbaWinTieLoss is True
    order_stats : str, default = 'average-statistic', the way to order the used_statistic, default
                  setup orders by average statistic over all datasets
    order_better : str, default = 'decreasing', by which order to sort stats, from best to worse
    dataset_column : str, default = 'dataset_name', the name of the datasets column in the csv file

    Returns
    -------
    analysis : python dictionary containing all extracted comparisons

    """

    analysis = {
            'dataset-column' : dataset_column,
            'used-mean' : used_mean,
            'order-stats' : order_stats,
            'order-better' : order_better,
            'used-statistics' : used_statistic,
            'order-WinTieLoss' : order_WinTieLoss,
            'include-pvalue' : include_pvalue,
            'pvalue-test' : pvalue_test
        }

    df_results = pd.read_csv(df_results_path)
    decode_results_data_frame(df=df_results, analysis=analysis)

    if order_stats == 'average-statistic':
        average_statistic = {}

    for i in range(analysis['n-classifiers']):

        if order_stats == 'average-statistic':
            average_statistic[analysis['classifier-names'][i]] = round(np.mean(df_results[analysis['classifier-names'][i]]), 4)

        for j in range(analysis['n-classifiers']):

            if i != j:

                print(analysis['classifier-names'][i],' vs ', analysis['classifier-names'][j])

                x = df_results[analysis['classifier-names'][i]]
                y = df_results[analysis['classifier-names'][j]]
                
                pairwise_content = get_pairwise_content(x=x,y=y,
                                                        order_WinTieLoss=order_WinTieLoss,
                                                        includeProbaWinTieLoss=include_ProbaWinTieLoss,
                                                        include_pvalue=include_pvalue,
                                                        pvalue_test=pvalue_test,
                                                        pvalue_threshhold=pvalue_threshhold,
                                                        used_mean=used_mean,
                                                        bayesian_rope=bayesian_rope)
                
                analysis[analysis['classifier-names'][i]+'-vs-'+analysis['classifier-names'][j]] = pairwise_content
                
                if plot_1v1_comparisons:
                    plot_1v1(x=x,y=y,
                            name_x=analysis['classifier-names'][i],
                            name_y=analysis['classifier-names'][j],
                            win_x=pairwise_content['win'],
                            tie=pairwise_content['tie'],
                            loss_x=pairwise_content['loss'],
                            output_directory=output_dir)


    analysis['average-statistic'] = average_statistic

    if pvalue_correction == 'Holm':

        pvalues = []

        for i in range(analysis['n-classifiers']):
            for j in range(i+1, analysis['n-classifiers']):
                if i != j:
                    pvalues.append(analysis[analysis['classifier-names'][i]+'-vs-'+analysis['classifier-names'][j]]['pvalue'])
    
        pvalues_sorted = np.sort(pvalues)

        k = 0
        m = len(pvalues)

        pvalue_times_used = {}

        for pvalue in pvalues:
            pvalue_times_used[pvalue] = 0

        for i in range(analysis['n-classifiers']):
            for j in range(i+1, analysis['n-classifiers']):
                if i !=  j:

                    pvalue = analysis[analysis['classifier-names'][i]+'-vs-'+analysis['classifier-names'][j]]['pvalue']
                    index_pvalue = np.where(pvalues_sorted == pvalue)[0]

                    if len(index_pvalue) == 1:
                        index_pvalue = index_pvalue[0]
                    else:
                        index_pvalue = index_pvalue[pvalue_times_used[pvalue]]
                        pvalue_times_used[pvalue] += 1

                    pvalue_threshhold_corrected = pvalue_threshhold / (m-index_pvalue)

                    if pvalue < pvalue_threshhold_corrected:
                        analysis[analysis['classifier-names'][i]+'-vs-'+analysis['classifier-names'][j]]['is-significant'] = True
                    else:
                        analysis[analysis['classifier-names'][i]+'-vs-'+analysis['classifier-names'][j]]['is-significant'] = False
                    
                    k = k + 1
        
        for i in range(analysis['n-classifiers']):
            for j in range(i+1,analysis['n-classifiers']):
                
                analysis[analysis['classifier-names'][j]+'-vs-'+analysis['classifier_names'][i]]['is-significant'] = analysis[analysis['classifier-names'][i]+'-vs-'+analysis['classifier-names'][j]]['is-significant']
                analysis[analysis['classifier-names'][j]+'-vs-'+analysis['classifier_names'][i]]['pvalue'] = analysis[analysis['classifier-names'][i]+'-vs-'+analysis['classifier-names'][j]]['pvalue']

    re_order_classifiers(df_results=df_results,
                        analysis=analysis)

    if save_as_json:
        with open(output_dir + 'analysis.json', 'w') as fjson:
            json.dump(analysis, fjson, cls=NpEncoder)
    
    return analysis

def get_heatmap(analysis=None,
                output_dir='./',
                load_analysis=True,
                colormap='coolwarm',
                fig_size='auto',
                font_size=17,
                pixels_per_clf_hieght=3,
                pixels_per_clf_width=3.5,
                show_symetry=True):
    
    """
    
    Draw the heatmap 1v1 multi classifier comparison

    Parameters
    ----------
    analysis : python dict, default = None, a python dictionary exrtracted using get_analysis function
    output_dir : str, default = './', output directory for the results
    load_analysis : bool, default = True, whether or not to load the analysis json file
    colormap : str, default = 'coolwarm', the colormap used in matplotlib
    fig_size : str ot tuple of two int, default = 'auto', the height and width of the figure,
               if 'auto', use get_fig_size function in utils.py
    font_size : int, default = 17, the font size of text
    pixels_per_clf_hieght : float, default = 3, the number of pixels used per classifier in height
                            inside each cell of the heatmap
    pixels_per_clf_width : float, default = 3.5, the number of pixels used per classifier in width
                           inside each cell of the heatmap
    show_symetry : bool, default = True, whether or not to show the symetrical part of the heatmap
    
    """

    if analysis is None:
        
        if load_analysis:
            with open(output_dir + 'analysis.json') as json_file:
                analysis = json.load(json_file)
        
        else:
            raise ValueError("If no analysis dictionary is prvided then the argument load_analysis should be true")

    pairwise_matrix = np.zeros(shape=(analysis['n-classifiers'], analysis['n-classifiers']))
    df_annotations = pd.DataFrame(columns=['Classifier']+[analysis['ordered-classifier-names'][i] for i in range(analysis['n-classifiers'])])

    start_index = 0

    string_to_add = ''
    string_to_add = string_to_add + analysis['used-mean'] + '\n'
    string_to_add = string_to_add + 'Win/Tie/Loss ' + analysis['order-WinTieLoss'] + '\n'
    if analysis['include-pvalue']:
        string_to_add = string_to_add + analysis['pvalue-test'] + ' p-value'

    for i in range(analysis['n-classifiers']):


        dict_to_add = {'Classifier' : analysis['ordered-classifier-names'][i]}

        if i == 0:
            dict_to_add[analysis['ordered-classifier-names'][i]] = string_to_add

        if not show_symetry:
            start_index = i + 1

        for j in range(start_index, analysis['n-classifiers']):

            if i != j:

                n_info_per_line = 0
                pairwise_matrix[i,j] = analysis[analysis['ordered-classifier-names'][i]+'-vs-'+analysis['ordered-classifier-names'][j]]['mean']
                
                pairwise_content = analysis[analysis['ordered-classifier-names'][i]+'-vs-'+analysis['ordered-classifier-names'][j]]
                pairwise_keys = list(pairwise_content.keys())

                string_to_add = ''

                string_to_add = string_to_add + str(round(pairwise_content['mean'],4))
                string_to_add = string_to_add + '\n'
                n_info_per_line += 1

                if 'win' in pairwise_keys:

                    string_to_add = string_to_add + str(pairwise_content['win']) + ' / '
                    string_to_add = string_to_add + str(pairwise_content['tie']) + ' / '
                    string_to_add = string_to_add + str(pairwise_content['loss'])
                    string_to_add = string_to_add + '\n'
                    n_info_per_line += 1

                if 'p_x_wins' in pairwise_keys:

                    string_to_add = string_to_add + str(round(pairwise_content['p-x-wins'],4)) + ' / '
                    string_to_add = string_to_add + str(round(pairwise_content['p-rope'],4)) + ' / '
                    string_to_add = string_to_add + str(round(pairwise_content['p-y-wins'],4))
                    string_to_add = string_to_add + '\n'
                    n_info_per_line += 1
                
                if 'pvalue' in pairwise_keys:

                    string_to_add = string_to_add 
                    string_to_add = string_to_add + str(round(pairwise_content['pvalue'],4))
                    n_info_per_line += 1

                dict_to_add[analysis['ordered-classifier-names'][j]] = string_to_add

        if not show_symetry:
            for j in range(i+1):

                dict_to_add[analysis['ordered-classifier-names'][j]] = '-'
                pairwise_matrix[i,j] = 0.0
        
        if show_symetry:
            pairwise_matrix[i,i] = 0.0
            if i > 0:
                dict_to_add[analysis['ordered-classifier-names'][i]] = '-'

        df_annotations = df_annotations.append(dict_to_add, ignore_index=True)
    
    df_annotations.to_csv(output_dir + 'heatmap.csv', index=False)

    df_annotations.drop('Classifier', inplace=True, axis=1)
    df_annotations_np = np.asarray(df_annotations)

    figsize = get_fig_size(fig_size=fig_size,
                           n_classifiers=analysis['n-classifiers'],
                           pixels_per_clf_width=pixels_per_clf_width,
                           pixels_per_clf_hieght=pixels_per_clf_hieght,
                           n_info_per_line=n_info_per_line)

    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]))

    min_value, max_value = get_limits(pairwise_matrix=pairwise_matrix)

    im = ax.imshow(pairwise_matrix,
                   cmap=colormap,
                   aspect='auto',
                   vmin=min_value + 0.2*min_value,
                   vmax=max_value + 0.2*max_value)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label(label=analysis['used-mean'], size=font_size, weight='bold')

    ticks = get_ticks(analysis)

    ax.set_xticks(np.arange(analysis['n-classifiers']), labels=ticks, fontsize=font_size)
    ax.set_yticks(np.arange(analysis['n-classifiers']), labels=ticks, fontsize=font_size)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.spines[:].set_visible(False)

    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              fontsize=font_size)

    for i in range(analysis['n-classifiers']):
        for j in range(analysis['n-classifiers']):
    
            if 'pvalue' in pairwise_keys:
                if i != j and analysis[analysis['ordered-classifier-names'][i]+'-vs-'+analysis['ordered-classifier-names'][j]]['is-significant']:
                    kw.update(fontweight='bold')

            im.axes.text(j, i, df_annotations_np[i, j], **kw)
            kw.update(fontweight='normal')

    plt.savefig(output_dir + 'heatmap.pdf')
    plt.cla()
    plt.clf()
    plt.close()

def get_line_heatmap(proposed_method,
                     analysis=None,
                     output_dir='./',
                     load_analysis=True,
                     colormap='coolwarm',
                     fig_size='auto',
                     font_size=17,
                     pixels_per_clf_hieght=10,
                     pixels_per_clf_width=3):

    """
    
    Draw the heatline 1v1 multi classifier comparison of a proposed_method vs all other methods

    Parameters
    ----------
    proposed_method : str, the proposed mehtod's name
    analysis : python dict, default = None, a python dictionary exrtracted using get_analysis function
    output_dir : str, default = './', output directory for the results
    load_analysis : bool, default = True, whether or not to load the analysis json file
    colormap : str, default = 'coolwarm', the colormap used in matplotlib
    fig_size : str ot tuple of two int, default = 'auto', the height and width of the figure,
               if 'auto', use get_fig_size function in utils.py
    font_size : int, default = 17, the font size of text
    pixels_per_clf_hieght : float, default = 10, the number of pixels used per classifier in height
                            inside each cell of the heatline
    pixels_per_clf_width : float, default = 3, the number of pixels used per classifier in width
                           inside each cell of the heatline
    
    """

    plt.cla()
    plt.clf()

    if analysis is None:
        
        if load_analysis:
            with open(output_dir + 'analysis.json') as json_file:
                analysis = json.load(json_file)
        
        else:
            raise ValueError("If no analysis dictionary is prvided then the argument load_analysis should be true")
    
    names_classifiers = []
    ordered_stats = []

    for i in range(analysis['n-classifiers']):

        names_classifiers.append(analysis['ordered-classifier-names'][i])
        ordered_stats.append(analysis['ordered-stats'][i])

    pairwise_line = np.zeros(shape=(1, analysis['n-classifiers']))
    df_annotations = pd.DataFrame(columns=['Classifier']+names_classifiers)

    dict_to_add = {'Classifier' : proposed_method}

    for i in range(len(names_classifiers)):

        if names_classifiers[i] == proposed_method:

            string_to_add = ''
            string_to_add = string_to_add + analysis['used-mean'] + '\n'
            string_to_add = string_to_add + 'Win/Tie/Loss ' + analysis['order-WinTieLoss'] + '\n'
            if analysis['include-pvalue']:
                string_to_add = string_to_add + analysis['pvalue-test'] + ' p-value'

            pairwise_line[0,i] = 0.0
            dict_to_add[names_classifiers[i]] = string_to_add

        else:

            n_info_per_line = 0

            pairwise_line[0,i] = analysis[proposed_method+'-vs-'+names_classifiers[i]]['mean']
                    
            pairwise_content = analysis[proposed_method+'-vs-'+names_classifiers[i]]
            pairwise_keys = list(pairwise_content.keys())

            string_to_add = ''

            string_to_add = string_to_add + str(round(pairwise_content['mean'],4))
            string_to_add = string_to_add + '\n'
            n_info_per_line += 1

            if 'win' in pairwise_keys:

                string_to_add = string_to_add + str(pairwise_content['win']) + ' / '
                string_to_add = string_to_add + str(pairwise_content['tie']) + ' / '
                string_to_add = string_to_add + str(pairwise_content['loss'])
                string_to_add = string_to_add + '\n'
                n_info_per_line += 1
            
            if 'p_x_wins' in pairwise_keys:

                string_to_add = string_to_add + str(round(pairwise_content['p-x-wins'],4)) + ' / '
                string_to_add = string_to_add + str(round(pairwise_content['p-rope'],4)) + ' / '
                string_to_add = string_to_add + str(round(pairwise_content['p-y-wins'],4))
                string_to_add = string_to_add + '\n'
                n_info_per_line += 1
            
            if 'pvalue' in pairwise_keys:

                string_to_add = string_to_add + str(round(pairwise_content['pvalue'],4))
                n_info_per_line += 1

            dict_to_add[names_classifiers[i]] = string_to_add

    df_annotations = df_annotations.append(dict_to_add, ignore_index=True)

    df_annotations.drop('Classifier', inplace=True, axis=1)
    df_annotations_np = np.asarray(df_annotations)

    figsize = get_fig_size(fig_size=fig_size,
                           n_classifiers=len(names_classifiers),
                           pixels_per_clf_width=pixels_per_clf_width,
                           pixels_per_clf_hieght=pixels_per_clf_hieght,
                           n_info_per_line=n_info_per_line)

    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]))

    min_value, max_value = get_limits(pairwise_matrix=pairwise_line)

    im = ax.imshow(pairwise_line,
                   cmap=colormap,
                   aspect='auto',
                   vmin=min_value + 0.20*min_value,
                   vmax=max_value + 0.2*max_value)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label(label=analysis['used-mean'], size=font_size*0.7, weight='bold')

    xticks, yticks = get_ticks_heatline(analysis=analysis, proposed_method=proposed_method)

    ax.set_xticks(np.arange(len(names_classifiers)), labels=xticks, fontsize=font_size)
    ax.set_yticks(np.arange(1), labels=yticks, fontsize=font_size)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.spines[:].set_visible(False)

    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              fontsize=font_size)

    for i in range(len(names_classifiers)):

        if names_classifiers[i] != proposed_method:

            if 'pvalue' in pairwise_keys:
                if analysis[proposed_method+'-vs-'+names_classifiers[i]]['is-significant']:
                    kw.update(fontweight='bold')

        im.axes.text(i, 0, df_annotations_np[0, i], **kw)
        kw.update(fontweight='normal')

    plt.savefig(output_dir + proposed_method+'_heatline.pdf')

if __name__ == "__main__":

    path_res = 'results.csv'

    analysis = get_analysis(df_results_path=path_res, save_as_json=True)
    get_heatmap()