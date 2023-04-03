import pandas as pd

from MCM import MCM

if __name__ == "__main__":

    path_res = './results_ucr2.csv'
    output_dir = './'

    df_results = pd.read_csv(path_res)

    MCM.compare(
        df_results=df_results,
        show_symetry=False,
        excluded_row_comparates=['ROCKET']
    )