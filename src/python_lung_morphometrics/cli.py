"""Console script for python_lung_morphometrics."""
from python_lung_morphometrics import do_mli as _do_mli

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
    save_dir: str = "output",
    min_chord_length: float = 3.0,
    max_chord_length: float = 500.0,
    max_image_length: int = 3500
):

    '''
    Given the filename of an H&E image,
    measure mean chord lengths. Prints results
    to console. 

    Output is to standard out: `filename\t MIL(um)`

    :meta private:
    '''

    ret = _do_mli(
        filename,
        save_pic,
        save_dir,
        min_chord_length,
        max_chord_length,
        max_image_length
    )

    console.print(
        "\"" + filename + "\"\t" + str(round(ret, 3)),
        soft_wrap=True
    )

    return None


if __name__ == "__main__":
    app()