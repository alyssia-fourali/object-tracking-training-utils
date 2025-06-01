==============================
object-tracking-training-utils
==============================


.. image:: https://img.shields.io/pypi/v/object_tracking_training_utils.svg
        :target: https://pypi.python.org/pypi/object_tracking_training_utils

.. image:: https://img.shields.io/travis/alyssia-fourali/object_tracking_training_utils.svg
        :target: https://travis-ci.com/alyssia-fourali/object_tracking_training_utils

.. image:: https://readthedocs.org/projects/object-tracking-training-utils/badge/?version=latest
        :target: https://object-tracking-training-utils.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




A Python toolkit for training and evaluating object tracking models


* Free software: MIT license
* Documentation: https://object-tracking-training-utils.readthedocs.io.


Features
--------

- I/O utilities for handling annotations, checkpoints, and logs
- Image preprocessing: resizing, cropping, padding, normalization, format conversion
- Training utilities: running average, timer, parameter loading and saving
- Plotting tools for histograms, learning curves, and debugging visualizations
- Visualization of frames with bounding boxes, response maps, and tracking outputs

Installation
------------
 install the latest version directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/alyssia-fourali/object-tracking-training-utils.git

Or clone and install in editable mode:

.. code-block:: bash

    git clone https://github.com/alyssia-fourali/object-tracking-training-utils.git
    cd object-tracking-training-utils
    pip install -e .

Usage
-----

Basic example:

.. code-block:: python

    from object_tracking_training_utils.visualization import show_frame
    show_frame(frame, bbox=[100, 50, 80, 60], fig_n=1, pause=2)


Contributing
------------

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-xyz`)
3. Make your changes
4. Submit a pull request

Before submitting, please make sure to run tests and follow the project's coding standards.

License
-------

This project is licensed under the MIT License - see the `LICENSE` file for details.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
