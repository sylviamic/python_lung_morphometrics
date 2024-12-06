=====
Usage
=====

To use python lung morphometrics in a project::

    import python_lung_morphometrics

To use the command-line script:

* Measure the mean linear intercept (MLI) of one image and print results to console::
    python-lung-morphometrics do-mli image_001.tif

* Measure the MLI of many images and save the results::
    ls -1 ./image_*.tif | xargs -I {} python-lung-morphometrics do-mli {} > ./results.tsv

* Measure the MLI of many images in parallel and save the results::
    parallel -j5 --bar python-lung-morphometrics do-mli ::: ./image_*.tif > ./results.tsv
