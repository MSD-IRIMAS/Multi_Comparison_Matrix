import pandas as pd

from MCM import MCM

if __name__ == "__main__":

    path_res = './results.csv'
    output_dir = './'

    df_results = pd.read_csv(path_res)

    analysis = MCM.get_analysis(df_results=df_results,
                                     save_as_json=True,
                                     plot_1v1_comparisons=False,
                                     output_dir=output_dir)
    
    MCM.get_heatmap(output_dir=output_dir,
                         colormap='coolwarm',
                         show_symetry=True)

    MCM.get_line_heatmap(proposed_methods=['ROCKET','ResNet'],
                         disjoint_methods=True,
                         df_results=df_results,
                         output_dir=output_dir)
