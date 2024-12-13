"""Console script for python_lung_morphometrics."""
from .python_lung_morphometrics import do_mli as _do_mli

import typer
from rich.console import Console

app = typer.Typer()
console = Console()    

@app.command()
def main():
    '''
    Dummy function.
    :meta private:
    '''

    console.print("dummy output")

@app.command()
def do_mli(
    filename: str,
    save_pic: bool = True,
    save_chords: bool = False,
    save_dir: str = "output",
    min_chord_length: float = 2.0,
    max_chord_length: float = 500.0,
    max_image_length: int = 4000,
    lateral_resolution: float = None
):

    '''
    Given the filename of an H&E image,
    measure mean chord lengths. Prints results
    to console. 

    Output to console: `filename\t MLI(um)`
    '''

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


if __name__ == "__main__":
    app()