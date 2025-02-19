=========================
python lung morphometrics
=========================

A python module for performing morphometric measurements
on microscopy images of lung tissue. 

Status
------

.. image:: https://woodpecker.sylviamichki.xyz/api/badges/1/status.svg
   :target: https://woodpecker.sylviamichki.xyz/repos/1

Run it in a container
---------------------

.. code-block:: console

    $ docker run -v ./data:/data docker.io/sylviamic/python_lung_morphometrics:latest do-mli /data/image_001.tif


Installation
------------

.. code-block:: console

    $ git clone git://github.com/sylviamic/python_lung_morphometrics
    $ cd python_lung_morphometrics
    $ pip install -U .


CLI usage
---------

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

Credits
-------

This package was created by Sylvia N. Michki.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

License
-------

Free software: GNU General Public License v3