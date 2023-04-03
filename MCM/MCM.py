import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import json

from tqdm import tqdm

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

def compare(
        df_results,
        output_dir='./',
        fig_savename="",
        used_statistic='Accuracy',
        save_as_json=True,
        plot_1v1_comparisons=False,
        order_WinTieLoss='higher',
        include_ProbaWinTieLoss=False,
        bayesian_rope=0.01,
        include_pvalue=True,
        pvalue_test='wilcoxon',
        pvalue_correction=None,
        pvalue_threshold=0.05,
        use_mean='mean-difference',
        order_stats='average-statistic',
        order_better='decreasing',
        dataset_column=None,
        precision=4,
        load_analysis=True,

        row_comparates=None,
        col_comparates=None,
        excluded_row_comparates=None,
        excluded_col_comparates=None,
        colormap='coolwarm',
        fig_size='auto',
        font_size='auto',
        colorbar_orientation='vertical',
        colorbar_value=None,
        win_label="row > col",
        tie_label="row = col",
        loss_label="row < col",
        include_legend=True,
        show_symetry=True,
):
    if isinstance(df_results, str):
        # assuming its a path
        try:
            df_results = pd.read_csv(df_results)
        except:
            print("No dataframe or valid path is given")
            return

    analysis = get_analysis(
        df_results,
        output_dir=output_dir,
        used_statistic=used_statistic,
        save_as_json=save_as_json,
        plot_1v1_comparisons=plot_1v1_comparisons,
        order_WinTieLoss=order_WinTieLoss,
        include_ProbaWinTieLoss=include_ProbaWinTieLoss,
        bayesian_rope=bayesian_rope,
        include_pvalue=include_pvalue,
        pvalue_test=pvalue_test,
        pvalue_correction=pvalue_correction,
        pvalue_threshhold=pvalue_threshold,
        use_mean=use_mean,
        order_stats=order_stats,
        order_better=order_better,
        dataset_column=dataset_column,
        precision=precision,
        load_analysis=load_analysis,
    )

    #### start drawing heatmap
    draw(
        analysis,
        savename=fig_savename,
        output_dir=output_dir,
        row_comparates=row_comparates,
        col_comparates=col_comparates,
        excluded_row_comparates=excluded_row_comparates,
        excluded_col_comparates=excluded_col_comparates,
        precision=precision,

        colormap=colormap,
        fig_size=fig_size,
        font_size=font_size,
        colorbar_orientation=colorbar_orientation,
        colorbar_value=colorbar_value,
        win_label=win_label,
        tie_label=tie_label,
        loss_label=loss_label,
        include_legend=include_legend,
        show_symetry=show_symetry,
    )

def get_analysis(df_results,
                 output_dir='./',
                 used_statistic='Score',
                 save_as_json=True,
                 plot_1v1_comparisons=False,
                 order_WinTieLoss='higher',
                 include_ProbaWinTieLoss=False,
                 bayesian_rope=0.01,
                 include_pvalue=True,
                 pvalue_test='wilcoxon',
                 pvalue_correction=None,
                 pvalue_threshhold=0.05,
                 use_mean='mean-difference',
                 order_stats='average-statistic',
                 order_better='decreasing',
                 dataset_column=None,
                 precision=4,
                 load_analysis=True):
    
    """
    
    Get analysis of all the pairwise and multi comparate comparisons and store them in analysis
    python dictionary. With a boolean parameter, you can plot the 1v1 scatter results.

    Parameters
    ----------

    df_results : pandas DataFrame, the csv file containing results
    output_dir : str, default = './', the output directory for the results
    used_statistic : str, default = 'Score', one can imagine using error, time, memory etc. instead
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
    used_mean : str, default = 'mean-difference', the mean used to comapre two comparates.
    bayesian_rope : float, default = 0.01, the rope used in case include_ProbaWinTieLoss is True
    order_stats : str, default = 'average-statistic', the way to order the used_statistic, default
                  setup orders by average statistic over all datasets
    order_better : str, default = 'decreasing', by which order to sort stats, from best to worse
    dataset_column : str, default = 'dataset_name', the name of the datasets column in the csv file

    Returns
    -------
    analysis : python dictionary containing all extracted comparisons

    """

    save_file = output_dir + 'analysis.json'

    if load_analysis and os.path.exists(save_file):

        with open(save_file) as json_file:
            analysis = json.load(json_file)

        return analysis

    analysis = {
            'dataset-column' : dataset_column,
            'use-mean' : use_mean,
            'order-stats' : order_stats,
            'order-better' : order_better,
            'used-statistics' : used_statistic,
            'order-WinTieLoss' : order_WinTieLoss,
            'include-pvalue' : include_pvalue,
            'pvalue-test' : pvalue_test,
            'pvalue-threshold' : pvalue_threshhold,
            'pvalue-correction' : pvalue_correction
        }

    decode_results_data_frame(df=df_results, analysis=analysis)

    if order_stats == 'average-statistic':
        average_statistic = {}

    pbar = tqdm(range(analysis["n-comparates"]))

    for i in range(analysis['n-comparates']):

        comparate_i = analysis['comparate-names'][i]

        if order_stats == 'average-statistic':
            average_statistic[comparate_i] = round(np.mean(df_results[comparate_i]), precision)

        for j in range(analysis['n-comparates']):

            if i != j:

                comparate_j = analysis['comparate-names'][j]

                pbar.set_description(f"Processing {comparate_i}, {comparate_j}")
                
                pairwise_key = get_keys_for_two_comparates(comparate_i, comparate_j)

                x = df_results[comparate_i]
                y = df_results[comparate_j]
                
                pairwise_content = get_pairwise_content(
                    x=x,y=y,
                    order_WinTieLoss=order_WinTieLoss,
                    includeProbaWinTieLoss=include_ProbaWinTieLoss,
                    include_pvalue=include_pvalue,
                    pvalue_test=pvalue_test,
                    pvalue_threshhold=pvalue_threshhold,
                    use_mean=use_mean,
                    bayesian_rope=bayesian_rope)
                
                analysis[pairwise_key] = pairwise_content
                
                if plot_1v1_comparisons:

                    max_lim = max(x.max(), y.max())
                    min_lim = min(x.min(), y.min())

                    if max_lim < 1:

                        max_lim = 1
                        min_lim = 0

                    plot_1v1(
                        x=x,y=y,
                        name_x=comparate_i,
                        name_y=comparate_j,
                        win_x=pairwise_content['win'],
                        tie=pairwise_content['tie'],
                        loss_x=pairwise_content['loss'],
                        output_directory=output_dir,
                        max_lim=max_lim,
                        min_lim=min_lim,
                    )


    if order_stats == 'average-statistic':
        analysis['average-statistic'] = average_statistic

    if pvalue_correction == 'Holm':
        holms_correction(analysis=analysis)

    re_order_comparates(df_results=df_results, analysis=analysis)

    if save_as_json:
        with open(save_file, 'w') as fjson:
            json.dump(analysis, fjson, cls=NpEncoder)
    
    return analysis

def draw(analysis,
        output_dir='./',
        savename="",
        row_comparates=None,
        col_comparates=None,
        excluded_row_comparates=None,
        excluded_col_comparates=None,
        precision=4,

        colormap='coolwarm',
        fig_size='auto',
        font_size='auto',
        colorbar_orientation='vertical',
        colorbar_value=None,
        win_label='row > col',
        tie_label='row = col',
        loss_label='row < col',
        show_symetry=True,
        include_legend=True,
    ):
    
    """
    
    Draw the heatmap 1v1 multi comparate comparison

    Parameters
    ----------
    analysis : python dict, default = None, a python dictionary exrtracted using get_analysis function
    output_dir : str, default = './', output directory for the results
    load_analysis : bool, default = True, whether or not to load the analysis json file
    colormap : str, default = 'coolwarm', the colormap used in matplotlib, if set to None,
                    no color map is used and the heatmap is turned off, no colors will be seen
    colorbar_value : str, default = 'mean-difference', the values for which the heat map colors
                        are based on
    fig_size : str ot tuple of two int, default = 'auto', the height and width of the figure,
               if 'auto', use get_fig_size function in utils.py
    font_size : int, default = 17, the font size of text
    pixels_per_clf_hieght : float, default = 3, the number of pixels used per comparate in height
                            inside each cell of the heatmap
    pixels_per_clf_width : float, default = 3.5, the number of pixels used per comparate in width
                           inside each cell of the heatmap
    show_symetry : bool, default = True, whether or not to show the symetrical part of the heatmap
    colorbar_orientation : str, default = 'vertical', in which orientation to show the colorbar
                           either horizontal or vertical
    
    """

    if (col_comparates is not None) and (excluded_col_comparates is not None):
        print('Choose whether to include or exclude, not both!')
        return

    if (row_comparates is not None) and (excluded_row_comparates is not None):
        print('Choose whether to include or exclude, not both!')
        return

    if row_comparates is None:
        row_comparates = analysis['ordered-comparate-names']
    else:
        # order comparates
        row_comparates = [x for x in analysis['ordered-comparate-names'] if x in row_comparates]
    
    if col_comparates is None:
        col_comparates = analysis['ordered-comparate-names']
    else:
        col_comparates = [x for x in analysis['ordered-comparate-names'] if x in col_comparates]
    
    if excluded_row_comparates is not None:
        row_comparates = [x for x in analysis['ordered-comparate-names'] if not (x in excluded_row_comparates)]

    if excluded_col_comparates is not None:
        col_comparates = [x for x in analysis['ordered-comparate-names'] if not (x in excluded_col_comparates)]

    n_rows = len(row_comparates)
    n_cols = len(col_comparates)

    can_be_symmetrical = False

    if n_rows == n_cols == len(analysis['ordered-comparate-names']):
        can_be_symmetrical = True

    if n_rows == n_cols == 1:

        figure_aspect = 'equal'
        colormap = None

        if row_comparates[0] == col_comparates[0]:

            print(f'Row and Column comparates are the same, {row_comparates[0]}!')
            return
    else:
        figure_aspect = 'auto'
    
    if (n_rows == 1) and (n_cols == 2):
        colorbar_orientation = 'horizontal'
    
    elif (n_rows ==2) and (n_cols == 2):
        colorbar_orientation = 'vertical'
    
    elif (n_rows == 2) and (n_cols == 1):
        colorbar_orientation = 'vertical'
    
    elif n_rows <= 2:
        colorbar_orientation = 'horizontal'

    if colorbar_orientation == 'horizontal':
        pixels_per_clf_hieght = 2.5
    
    if include_legend:
        
        cell_legend, longest_string = get_cell_legend(analysis, win_label=win_label, tie_label=tie_label, loss_label=loss_label)
        
        if analysis['include-pvalue']:

            p_value_text = f"If in bold, then\np-value $<$ {analysis['pvalue-threshold']:.2f}"

            if analysis['pvalue-correction'] is not None:

                correction = capitalize_label(analysis['pvalue-correction'])
                p_value_text = f"{p_value_text}\n{correction} correction"
    
        else:
            p_value_text = ""
        
        longest_string = max(longest_string, len(p_value_text))
    
    else:
        cell_legend = ""
        p_value_text = ""
        longest_string = len(f"{win_label} / {tie_label} / {loss_label}")
    
    annot_out = get_annotation(
        analysis=analysis,
        row_comparates=row_comparates,
        col_comparates=col_comparates,
        cell_legend=cell_legend,
        p_value_text=p_value_text,
        colormap=colormap,
        colorbar_value=colorbar_value,
        precision=precision
    )

    df_annotations = annot_out["df_annotations"]
    pairwise_matrix = annot_out["pairwise_matrix"]

    n_info_per_cell = annot_out["n_info_per_cell"]

    legend_cell_location = annot_out["legend_cell_location"]
    p_value_cell_location = annot_out["p_value_cell_location"]

    longest_string = max(annot_out["longest_string"], longest_string)

    if savename != "":
        # todo: can add a argument to save or not
        df_annotations.to_csv(output_dir + f'{savename}.csv', index=False)
    
    df_annotations.drop('comparates', inplace=True, axis=1)
    df_annotations_np = np.asarray(df_annotations)

    figsize = get_fig_size(
        fig_size=fig_size,
        n_rows=n_rows,
        n_cols=n_cols,
        n_info_per_cell=n_info_per_cell,
        longest_string=longest_string
    )

    if font_size == 'auto':

        if (n_rows <= 2) and (n_cols <= 2):
            font_size = 8
        else:
            font_size = 10
    
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]))

    _can_be_negative = False
    if colorbar_value is None or colorbar_value == 'mean-difference':
        _can_be_negative = True
    min_value, max_value = get_limits(
        pairwise_matrix=pairwise_matrix,
        can_be_negative=_can_be_negative
    )

    if colormap is None:
        _colormap = 'coolwarm'
        _vmin, _vmax = -2, 2
    else:
        _colormap = colormap
        _vmin = min_value + 0.2 * min_value
        _vmax = max_value + 0.2 * max_value

    if colorbar_value is None:
        _colorbar_value = capitalize_label('mean-difference')
    else:
        _colorbar_value = capitalize_label(colorbar_value)

    im = ax.imshow(
        pairwise_matrix,
        cmap=colormap,
        aspect=figure_aspect,
        vmin=_vmin,
        vmax=_vmax
    )

    if colormap is not None:

        if (p_value_cell_location is None) and (legend_cell_location is None) and (colorbar_orientation == 'horizontal'):
            shrink = 0.4
        else:
            shrink = 0.5
        
        cbar = ax.figure.colorbar(im, ax=ax, shrink=shrink, orientation=colorbar_orientation)
        cbar.ax.tick_params(labelsize=font_size)
        cbar.set_label(label=capitalize_label(_colorbar_value), size=font_size)

    xticks, yticks = get_ticks(analysis, row_comparates, col_comparates, precision)
    ax.set_xticks(np.arange(n_cols), labels=xticks, fontsize=font_size)
    ax.set_yticks(np.arange(n_rows), labels=yticks, fontsize=font_size)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.spines[:].set_visible(False)

    start_j = 0


    for i in range(n_rows):

        row_comparate = row_comparates[i]

        if can_be_symmetrical and (not show_symetry):
            start_j = i

        for j in range(start_j, n_cols):

            col_comparate = col_comparates[j]

            cell_text_arguments = dict(
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=font_size
            )

            if row_comparate == col_comparate:

                if p_value_cell_location is not None:

                    if (i == p_value_cell_location[0]) and (j == p_value_cell_location[1]):

                        cell_text_arguments.update(
                            fontweight='bold',
                            fontsize=font_size
                        )

                if legend_cell_location is not None:

                    if (i == legend_cell_location[0]) and (j == legend_cell_location[1]):

                        cell_text_arguments.update(
                            fontsize=font_size
                        )

                im.axes.text(j, i, df_annotations_np[i, j], **cell_text_arguments)
                continue

            pairwise_key = get_keys_for_two_comparates(row_comparate, col_comparate)

            pairwise_content = analysis[pairwise_key]
            pairwise_keys = list(pairwise_content.keys())

            if "pvalue" in pairwise_keys:
                if analysis[pairwise_key]["is-significant"]:
                    cell_text_arguments.update(fontweight='bold')

            im.axes.text(j, i, df_annotations_np[i, j], **cell_text_arguments)

    if analysis['order-stats'] == 'average-statistic':
        ordering = 'Mean-' + analysis['used-statistics']
    else:
        ordering = analysis['order-stats']
    if n_cols == n_rows == 1:
        # special case when 1x1
        x = ax.get_position().x0 - 1
        y = ax.get_position().y1 - 1.5
    else:
        x = ax.get_position().x0 - 0.8
        y = ax.get_position().y1 - 1.5

    im.axes.text(
        x, y,
        ordering,
        fontsize=font_size,
        horizontalalignment="center",
        verticalalignment="center"
    )

    if p_value_cell_location is None:

        x = 0
        y = n_rows

        if n_rows == n_cols == 1:
            y = 0.7
        elif (n_cols == 1) and (legend_cell_location is None):
            x = -0.5
        elif (n_rows == 1) and (n_cols <= 2) and (colorbar_orientation == "horizontal"):
            x = -0.5

        im.axes.text(
            x, y,
            p_value_text,
            fontsize=font_size,
            fontweight="bold",
            horizontalalignment="center",
            verticalalignment="center",
        )

    if legend_cell_location is None:

        x = n_cols - 1
        y = n_rows
        if n_rows == n_cols == 1:
            x = n_cols + 0.5
            y = 0

        elif (n_rows == 1) and (colorbar_orientation == "horizontal"):
            x = n_cols + 0.25
            y = 0

        elif n_cols == 1:
            x = 0.5

        im.axes.text(
            x, y,
            cell_legend,
            fontsize=font_size,
            horizontalalignment="center",
            verticalalignment="center"
        )

    if savename != "":
        plt.savefig(os.path.join(output_dir + f"{savename}.pdf"), bbox_inches="tight")
        plt.savefig(os.path.join(output_dir + f"{savename}.png"), bbox_inches="tight")
        plt.cla()
        plt.clf()
        plt.close()
    else:
        plt.show()