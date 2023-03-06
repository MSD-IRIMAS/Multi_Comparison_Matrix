from ALTAMCBE import ALTAMCBE

if __name__ == "__main__":

    path_res = './results_example.csv'
    output_dir = './'

    analysis = ALTAMCBE.get_analysis(df_results_path=path_res,
                                     save_as_json=True,
                                     plot_1v1_comparisons=True,
                                     output_dir=output_dir)
    
    ALTAMCBE.get_heatmap(output_dir=output_dir,
                         colormap='coolwarm',
                         show_symetry=True)

    ALTAMCBE.get_line_heatmap(proposed_method='clf3', output_dir=output_dir, colormap='coolwarm',pixels_per_clf_hieght=7)