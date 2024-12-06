=========================
python lung morphometrics
=========================


A set of tools for measuring MLI and the like.


Features
--------

* Measure the mean linear intercept (MLI) of one image and print results to console::

        python-lung-morphometrics do-mli image_001.tif

* Measure the MLI of many images and save the results::

        ls -1 ./image_*.tif | xargs -I {} python-lung-morphometrics do-mli {} > ./results.tsv

* Measure the MLI of many images in parallel and save the results::

        parallel -j5 --bar python-lung-morphometrics do-mli ::: ./image_*.tif > ./results.tsv``

Credits
-------

This package was created by Sylvia N. Michki.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

License
-------

Free software: GNU General Public License v3
