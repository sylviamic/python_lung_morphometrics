=========================
python lung morphometrics
=========================


.. image:: https://img.shields.io/pypi/v/python_lung_morphometrics.svg
        :target: https://pypi.python.org/pypi/python_lung_morphometrics

.. image:: https://img.shields.io/travis/nigeil/python_lung_morphometrics.svg
        :target: https://travis-ci.com/nigeil/python_lung_morphometrics

.. image:: https://readthedocs.org/projects/python-lung-morphometrics/badge/?version=latest
        :target: https://python-lung-morphometrics.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




A set of tools for measuring MLI and the like.


* Free software: GNU General Public License v3
* Documentation: https://python-lung-morphometrics.readthedocs.io.


Features
--------

* Measure the mean linear intercept (MLI) of one image and print results to console: 
``
python-lung-morphometrics do-mli image_001.tif
``
* Measure the MLI of many images and save the results: 
``
ls -1 ./image_*.tif | xargs -I {} python-lung-morphometrics do-mli {} > ./results.tsv
``
* Measure the MLI of many images in parallel and save the results: 
``
parallel -j5 --bar python-lung-morphometrics do-mli ::: ./image_*.tif > ./results.tsv
``

Credits
-------

This package was created by Sylvia Michki.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
