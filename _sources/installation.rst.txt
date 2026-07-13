.. _Installation:

Installation
************

.. note::
    CuBATS currently requires Python 3.10 to 3.12.

Pip Install
===========

.. note::
    We suggest installing CuBATS and its prerequisites inside a virtual environment to avoid conflicts with other packages. On macOS and Linux, we recommend using the `venv <https://docs.python.org/3/library/venv.html>`_ module. For Windows, we recommend using `Conda <https://docs.conda.io/projects/conda/en/latest/user-guide/index.html>`_.

CuBATS can be downloaded from PyPI as the `cubats` package using the `pip` command. However, CuBATS also requires several system-level packages which need to be installed as well (see `Prerequisites`_ below). Once the prerequisites are installed, CuBATS can be installed via `pip`, ideally inside a previously created virtual environment as shown below:

On macOS and Linux:

.. code-block:: bash

    $ python3 -m venv cubats-env
    $ source cubats-env/bin/activate
    $ pip install cubats

On Windows:

.. code-block:: bash

    $ conda create -n cubats-env python=3.12
    $ conda activate cubats-env
    $ pip install cubats

Or install CuBATS directly from GitHub:

.. code-block:: bash

    $ pip install git+https://github.com/hnu-digihealth/CuBATS.git

.. _Prerequisites:

Prerequisites
=============

CuBATS relies on a few system level packages that need to be installed beforehand. Most importantly, `VALIS <https://valis.readthedocs.io/en/latest/>`_, `OpenSlide Python <https://openslide.org/api/python/>`_, `pyvips <https://libvips.github.io/libvips/>`_ and `scikit-image <https://scikit-image.org/docs/stable/>`_. Depending on the image format, `VALIS` may require additional tools for registration: While `OpenSlide` is sufficient for the image formats `.vmu`, `.mrxs`, and `.svslide`, for many other image formats the Java library `Bio-Formats <https://bio-formats.readthedocs.io>`_ is additionally needed.

pyvips
------

macOS
~~~~~

Install pyvips using Homebrew:

.. code-block:: bash

    $ brew install vips python pkg-config
    $ pip install pyvips

Ubuntu/Debian
~~~~~~~~~~~~~

Install pyvips using apt:

.. code-block:: bash

    $ sudo apt-get update
    $ sudo apt-get install libvips
    $ pip install pyvips

Windows
~~~~~~~

Download and install the precompiled binaries for libvips and follow the instructions for Windows on the official website: https://www.libvips.org/install.html.
Alternatively, you can install pyvips using vcpkg:

.. code-block:: bash

    $ git clone https://github.com/Microsoft/vcpkg.git
    $ cd vcpkg
    $ ./bootstrap-vcpkg.sh
    $ ./vcpkg integrate install
    $ ./vcpkg install vips
    $ pip install pyvips

OpenSlide Python
----------------

.. note:: that OpenSlide Python is a Python wrapper for OpenSlide. `OpenSlide <https://openslide.org>`_ is a C library that must to be installed separately.

macOS
~~~~~

Install OpenSlide and OpenSlide Python using Homebrew:

.. code-block:: bash

    $ brew install openslide
    $ pip install openslide-python

.. note::
    If you encounter issues with the OpenSlide library not being found, you may need to set the `DYLD_FALLBACK_LIBRARY_PATH` environment variable to point to the OpenSlide library directory. You can do this
    dynamically in your Python script or in your shell configuration file (e.g., `.bash_profile`, `.zshrc`, etc.):

    .. code-block:: python

        import os
        path = os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')
        path += (':' if path else '') + '/opt/homebrew/lib'
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = path
        import openslide

    Alternatively, you can set the `DYLD_LIBRARY_PATH` environment variable to point to the OpenSlide library directory:

    .. code-block:: bash

        export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:${DYLD_FALLBACK_LIBRARY_PATH}"

    After adding this line, reload your shell configuration:

    .. code-block:: bash

        $ source ~/.bash_profile
        # or
        $ source ~/.zshrc

Ubuntu/Debian
~~~~~~~~~~~~~

Install OpenSlide and OpenSlide Python using apt:

.. code-block:: bash

    $ sudo apt-get update
    $ sudo apt-get install openslide-tools
    $ pip install openslide-python

Windows
~~~~~~~

Download and install `OpenSlide binaries <https://openslide.org/download/#windows-binaries>`_ from the official website. Then import OpenSlide as follows [#f1]_:

.. code-block:: python

    OPENSLIDE_PATH = "path/to/openslide/bin"

    import os
    if hasattr(os, 'add_dll_directory'):
        with os.add_dll_directory(OPENSLIDE_PATH)
            import openslide
    else:
        import openslide


VALIS
-----

After having installed the previous prerequisites (optionally Bio-Formats), requirements for VALIS are met and the framework can be installed via pip:

.. code-block:: bash

    $ pip install valis-wsi

scikit-image
------------

scikit-image can be installed via pip:

.. code-block:: bash

    $ pip install scikit-image

or via conda:

.. code-block:: bash

    $ conda install scikit-image

Bio-Formats (Optional)
----------------------

Bio-Formats requires the Java Development Kit (JDK) and Maven. Install them as follows:

macOS
~~~~~

Install JDK and Maven using Homebrew:

.. code-block:: bash

    $ brew install openjdk maven

After installation, you may need to set the `JAVA_HOME` environment variable:

.. code-block:: bash

    export JAVA_HOME=$(/usr/libexec/java_home)

Ubuntu/Debian
~~~~~~~~~~~~~

Install JDK and Maven using apt:

.. code-block:: bash

    $ sudo apt-get update
    $ sudo apt-get install openjdk-11-jdk maven

After installation, you may need to set the `JAVA_HOME` environment variable:

.. code-block:: bash

    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

Windows
~~~~~~~

Download and install the JDK from the official website: https://www.oracle.com/java/technologies/javase-jdk11-downloads.html

Download and install Maven from the official website: https://maven.apache.org/download.cgi

After installation, you need to set the `JAVA_HOME` environment variable. For detailed instructions, refer to this guide: https://www.baeldung.com/java-home-on-windows-7-8-10

.. [#f1] This code is adapted from the `OpenSlide Python documentation <https://openslide.org/api/python/>`_.
