Installation
============

SkPM is available from Python 3.10 to 3.12.

Installation via PyPi
---------------------

SkPM will be available on PyPi soon!

.. .. code-block:: none

..    pip install skpm

Installation from source
------------------------

To install SkPM via GitHub, you can clone the repository and install it using pip:

.. code-block:: none

   git clone
   cd skpm
   pip install .

Alternatively, you can install it using poetry:

.. code-block:: none

   python3.10 -m venv skpm-venv
   source skpm-venv/bin/activate
   pip install -U pip setuptools poetry
   poetry install
