"""
Console script for python_lung_morphometrics.
"""

from ._do_mli import do_mli as _do_mli
from ._colocalization_analysis import do_colocalization_analysis as _do_colocalization_analysis

import typer
from rich.console import Console

app = typer.Typer()
console = Console()    

@app.command()
def main():
    """
    Dummy function.
    """

    console.print("dummy output")

@app.command()
def do_mli(
    filename: str,
    save_pic: bool = True,
    save_chords: bool = False,
    save_dir: str = "do_mli_output",
    min_chord_length: float = 2.0,
    max_chord_length: float = 500.0,
    max_image_length: int = 5000,
    lateral_resolution: float = None
):

    """
    Given the filename of an H&E image,
    measure mean chord lengths. Prints results
    to console. 

    Output to console: `filename\t MLI(um)`
    """

    ret = _do_mli(
        filename,
        save_pic,
        save_dir,
        save_chords,
        min_chord_length,
        max_chord_length,
        max_image_length, 
        lateral_resolution
    )

    console.print(
        "\"" + filename + "\"\t" + str(round(ret, 3)),
        soft_wrap=True
    )

    return None

@app.command()
def do_colocalization_analysis(
    filename: str,
    nuc_seg_img_filename: str = None,
    use_cellpose: bool = True,
    use_gpu: bool = False,
    channel_axis: int = 0,
    nucleus_channel_idx: int = 0,
    save_table: bool = True,
    save_intermediate_images: bool = True,
    dpi: int = 450,
    save_dir: str = "do_colocalization_analysis_output",
):

    """
    Given the filename of multi-channel TIFF,
    measure percent overlap of thresholded signal
    with segmented nuclear ROIs.

    Output to console: results table (pd.DataFrame)
    """

    ret = _do_colocalization_analysis(
        filename,
        nuc_seg_img_filename,
        use_cellpose,
        use_gpu,
        channel_axis,
        nucleus_channel_idx,
        save_table, 
        save_intermediate_images,
        dpi,
        save_dir
    )

    console.print(
        ret,
        soft_wrap=True
    )

    return None


if __name__ == "__main__":
    app()