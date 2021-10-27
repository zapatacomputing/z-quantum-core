=====================================================
How to Use the z-quantum-action Tools
=====================================================

Introduction
================

This repo contains definitions of standardized GitHub Actions, Makefile, and
Python configuration files used in ``z-quantum-*``, ``orquestra-*`` and ``qe-*``
repositories.

Repo contents
-------------

Files for Python development:

   * README.rst: This file
   * Makefile: Makefile to build, test, lint, etc... To be included in top level
   * pyproject.toml: A sample pyproject.toml file. Can be symlinked in top level
   * pytest.ini: A sample pytest.ini, can be symlinked
   * sample_setup.py: a sample setup.py file
   * setup_extras.py: extra imports for setup.py develop mode
   * variables.mk: Makefile global variables


Files for Github Actions:

   * workflow-templates/coverage.yml: Workflow to test code test coverage
   * workflow-templates/publish-release.yml: workflow to publish release
     workflow template
   * workflow-templates/style.yml: Workflow to test all style linters
   * workflow-templates/submit-dev-main-pr.yml
   * actions/: All actions folders for *uses:* includes for
     workflow-templates should not be modified normally.
   * actions/coverage/action.yml 
   * actions/publish-release/action.yml
   * actions/ssh_setup/action.yml
   * actions/style/action.yml

Actions Subtree Setup
==========================
This repo is designed to be subtree added to your git repo. 

.. Note:: We use underscores in the subtree folder name so that python imports
   to *subtrees.z_quantum_actions/* work properly.

Example::

   cd z-quantum-somerepo
   git subtree add -P subtrees/z_quantum_actions git@github.com:zapatacomputing/z-quantum-actions.git main --squash

Subtree Pull Updates (Recommended)
---------------------------------------
You have modified the subtree remotely. You'd like to pull those changes to the
composite repo::

    cd z-quantum-somerepo
    git subtree pull --prefix subtrees/z_quantum_actions git@github.com:zapatacomputing/z-quantum-actions.git main --squash

Subtree Push Updates (Not Recommended)
----------------------------------------

.. Note:: We don't recommend this method because its not reliable and can cause
   merge issues. Instead, consider making changes in the ``z-quantum-actions``
   repo directly.

You have modified the subtree locally in the "joined" branch. You'd like to push those
changes to the subtree's origin so that others can benefit from the fixes::

    git subtree push --prefix subtrees/z_quantum_actions git@github.com:zapatacomputing/zqs-python-env.git main


Github Actions Setup
-------------------------

There are several github actions in this subtree::

    workflows-templates/coverage.yml
    workflows-templates/publish-release.yml
    workflows-templates/style.yml
    workflows-templates/submit-dev-main-pr.yml

These can be configured as follows::

    cd z-quantum-somerepo
    mkdir -p .github/workflows
    cd .github/workflows/
    cp -a ../../subtrees/z_quantum_actions/workflow-templates/ .

.. Note:: These action templates depend on the include files located in:
   ``../../subtrees/z_quantum_actions/actions/*`` folders.

   Actions above will often referred to their local folder paths, like this:

   .. code-block:: bash

       # inside your GitHub workflow
       steps:
         - name: Run publish release action
           uses: ./subtrees/z_quantum_actions/publish-release



Python Configuration
==========================

Using the Makefile
--------------------------

Create an empty Makefile in the top directory with this one line::

    echo 'include subtrees/z_quantum_actions/Makefile' >> Makefile

If your Makefile already has targets in it, this will mask your custom
targets. Please ensure the added include is at the top of the file.

Overriding the Makefile
~~~~~~~~~~~~~~~~~~~~~~~~~
All Makefile targets can be overridden. In the top level Makefile, after the
include, just add your override. For example ::

    include subtrees/z_quantum_actions/Makefile

    test:
         pytest tests

Configuring setup.py
--------------------------

.. Note:: We have labeled the development target as *develop* to avoid confusion
   with the *dev* branch name.

* Backup your original setup.py
* Copy current sample_setup.py to your repo's setup.py in your target repo.

Inside of setup.py we need to make sure the extras are in place::

    try:
        from subtrees/z_quantum_actions.setup_extras import extras
    except ImportError:
        print("Unable to import extras")
        extras = {}
    else:
        print("Imported subtrees/z_quantum_actions.setup_extras.extras")

If you have other additions to extras_requirements, you need to add those in
manually before you call *setup()* ::

    extras.get('develop').append('my_requirement>=1.2.0')

Finally, inside of *setup()* set ::

    extras_require=extras,

Configuring pyproject.toml
---------------------------
Use the included pyproject.toml as a template for the build and style directives.
You can also soft-link the file to your top level folder if desired::

    cd ~/z-quantum-somerepo
    ln -s subtrees/z_quantum_actions/pyproject.toml .

If you have other requirements, those must be included as needed.

Pytest Configuraiton
=========================
The included Makefile allow us to simply do a *make test* and *make coverage*.
Because ``pytest.ini`` is fully supported (as compared with ``pyproject.toml``)
we use that for now. You can soft-link it similarly to before::

    cd ~/z-quantum-somerepo
    ln -s subtrees/z_quantum_actions/pytest.ini .

Marking Tests
--------------
We have included one Pytest mark in ``pytest.ini``:

* integration

You can apply the decorator at either the class, method, or function level::

    @pytest.mark.integration

To make life as easy as possible you can also mark an entire Pytest module file
like this::

    import pytest
    pytestmark = pytest.mark.integration

or for multiple markers::

    pytestmark = [pytest.mark.integration, pytest.mark.ufo]

By default integration tests are disabled. In order to enable them you can just define
   a new *test* target as *pytest tests* (removing the *-m integration*) and forget
   about all the markings.
