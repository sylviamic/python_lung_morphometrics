.. highlight:: shell

============
Installation
============

Using a container
-----------------

Run the CLI in a (docker) container. Be sure to mount 
the directory that holds your data into the container using the
appropriate command-line flag (:code:`-v ./your/local_path:/path_inside_container`):

.. code-block:: console

    $ docker run -v ./tests/data:/data docker.io/sylviamic/python_lung_morphometrics:latest do-mli /data/image_001.tif

From sources
------------

The sources for python lung morphometrics can be downloaded from the `Github repo`_.

.. code-block:: console

    $ git clone git://github.com/sylviamic/python_lung_morphometrics

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install -U -e .

This will force a local user installation in 'editable mode',
meaning changes made to source files will propagate without the 
need for re-installation. Useful for debugging and prototyping 
new functionality.


.. _Github repo: https://github.com/sylviamic/python_lung_morphometrics
.. _tarball: https://github.com/sylviamic/python_lung_morphometrics/tarball/master


Stable release (coming soon!)
-----------------------------

To install python lung morphometrics, run this command in your terminal:

.. code-block:: console

    $ pip install python_lung_morphometrics

This is the preferred method to install python lung morphometrics, as it will always install 
the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/