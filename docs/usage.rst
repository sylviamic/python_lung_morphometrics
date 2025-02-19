=====
Usage
=====

To use python lung morphometrics in a project::

    import python_lung_morphometrics

To use the command-line script:

Measure the mean linear intercept (MLI) of one image and print results to console:

.. code-block:: console

    $ python-lung-morphometrics do-mli image_001.tif

Measure the MLI of one image and save the test chords to a file in a specific directory:

.. code-block:: console

    $ python-lung-morphometrics do-mli --save-chords --save-dir ./my_output_dir image_001.tif

Measure the MLI of many images and save the results:

.. code-block:: console

    $ ls -1 ./image_*.tif | xargs -I {} python-lung-morphometrics do-mli {} > ./results.tsv

Measure the MLI of many images in parallel and save the results:

.. code-block:: console

    $ parallel -j5 --bar python-lung-morphometrics do-mli ::: ./image_*.tif > ./results.tsv``
