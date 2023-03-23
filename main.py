import pandas as pd

from MCM import MCM

if __name__ == "__main__":

    path_res = './results_example_no_data_column2.csv'
    output_dir = './'

    df_results = pd.read_csv(path_res)

    analysis = MCM.get_analysis(df_results=df_results,
                                     save_as_json=True,
                                     plot_1v1_comparisons=False,
                                     output_dir=output_dir,
                                     pvalue_correction='Holm')
    
    MCM.get_heatmap(output_dir=output_dir,
                         colormap='coolwarm',
                         show_symetry=True,
                         win_label='row>col',
                         tie_label='row=col',
                         loss_label='row<col')

    MCM.get_line_heatmap(proposed_methods=['clf1','clf2'],
                         output_dir=output_dir)
