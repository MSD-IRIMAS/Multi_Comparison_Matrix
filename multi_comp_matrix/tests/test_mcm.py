from multi_comp_matrix.MCM import compare
import pandas as pd
import tempfile
import time
import pytest

@pytest.mark.parametrize("save_type", ["pdf", "png", "tex"])
def test_all_type_saving(save_type):
    """Test each of the save types."""
    data = {
        'clf1': [1, 2, 3, 4],
        'clf2': [4.0, 6.0, 2.0, 1.0],
        'clf3': [10.5, 20.5, 30.5, 40.5]
    }
    df = pd.DataFrame(data)
    with tempfile.TemporaryDirectory() as tmp:
        if save_type == "pdf":
            curr_time = str(time.time_ns())
            
            compare(df_results=df, output_dir=tmp, pdf_savename=curr_time+"test_pdf")
            compare(df_results=df, output_dir=tmp, pdf_savename=curr_time+"test_vertical_pdf",
                    excluded_col_comparates=["clf1", "clf3"],)
            compare(df_results=df, output_dir=tmp, pdf_savename=curr_time+"test_horizontal_pdf",
                    excluded_row_comparates=["clf1", "clf3"],)
        elif save_type == "png":
            curr_time = str(time.time_ns())
            
            compare(df_results=df, output_dir=tmp, png_savename=curr_time+"test_png")
            compare(df_results=df, output_dir=tmp, png_savename=curr_time+"test_vertical_png",
                    excluded_col_comparates=["clf1", "clf3"],)
            compare(df_results=df, output_dir=tmp, png_savename=curr_time+"test_horizontal_png",
                    excluded_row_comparates=["clf1", "clf3"],)
        elif save_type == "tex":
            curr_time = str(time.time_ns())
            
            compare(df_results=df, output_dir=tmp, tex_savename=curr_time+"test_tex")
            compare(df_results=df, output_dir=tmp, tex_savename=curr_time+"test_vertical_tex",
                    excluded_col_comparates=["clf1", "clf3"],)
            compare(df_results=df, output_dir=tmp, tex_savename=curr_time+"test_horizontal_tex",
                    excluded_row_comparates=["clf1", "clf3"],)
            