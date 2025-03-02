Transport Frames
================

.. logo-start

.. figure:: https://sun9-46.userapi.com/impf/aUFBStH0x_6jN9UhgwrKN1WN4hZ9Y2HMMrXT2w/NuzVobaGlZ0.jpg?size=1590x400&quality=95&crop=0,0,1878,472&sign=9d33baa41a86de35d951d4bbd8011994&type=cover_group
   :alt: The Institute of Design and Urban Studies

.. logo-end

|Documentation Status| |PythonVersion| |Black| 

.. description-start

**Transport Frames** is a Python library designed for spatial transport analysis, offering tools to:

- Generate **transport frames** based on drive graphs.
- Identify **priority roads** by evaluating the expected popularity of routes between exit pairs.
- Update drive graphs with new road connections, ensuring proper integration.
- Grade **territory polygons** by analyzing distances to federal and regional roads.
- Compute a **weighted connectivity score** for a region based on accessibility to transport services and road network quality.
- Interpret accessibility scores into textual descriptions.
- Aggregate **administrative-level statistics** by computing transport indicators at different spatial scales (regions, districts, etc.).
- Analyze **territory-specific indicators** by using buffer-based calculations around the center of a given territory.

.. description-end

Features
--------

.. features-start


🚏 **Frame and Road Network Analysis**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Constructs **frames** from drive graphs to analyze road connectivity.
- Investigates **priority roads** based on estimated route popularity.
- Updates transport graphs by adding new edges and ensuring seamless network connectivity.

🌍 **Territory Grading and Connectivity**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Grades territories based on **proximity to federal and regional roads**.
- Computes **connectivity values** from road networks and accessibility to key transport services (bus stops, railways, airports, ports).
- Assigns **territory-wide scores** to summarize transport accessibility.

📊 **Indicator Computation**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The library calculates various transport indicators at different **administrative levels** (region, district, etc.), including:

- **Distance to regional centers** 
- **Distance to federal roads** 
- **Connectivity score**  (measuring how well a region is connected to the transport network)
- **Lengths of roads and railway paths** 
- **Road density**  (road length per area)
- **Service accessibility**  (bus routes, railway coverage, and general service accessibility)
- **Number of services**  (number of services inside or near a given territory)
- **Number of bus routes**  (number of bus routes intersecting the territory)
- **Distance to nature objects**  (water objects, nature reserves, etc.)

.. features-end

Installation
------------

.. installation-start

**Transport_frames** can be installed with ``pip``:

::

   pip install transport-frames

.. installation-end

How to use
----------

.. use-start

For the detailed tutorial on usage case see our `examples <#examples>`__.

The following **Jupyter Notebooks** illustrate core library functions:

1. **Graph Frame Creation** → `1_graph_frame_creation.ipynb <examples/1_graph_frame_creation.ipynb>`_
   - Created graph from territory polygon.
   - Generates transport frames from drive graphs.
   - Identifies priority roads using network analysis.

2. **Indicator Computation** → `2_indicators.ipynb <examples/2_indicators.ipynb>`_
   - Computes transport indicators (road density, connectivity, admin center distances, etc.).
   - Aggregates statistics at area and territory levels.

3. **Territory grading** → `3_criteria.ipynb <examples/3_criteria.ipynb>`_
   - Assigns territory scores based on federal and regional roads accessibility.
   - Analyzes proximity to key infrastructure (bus stops, ports, airports).
   - Analyzes connectivity metrics of the territory.
   - Converts numeric scores into textual interpretations.

4. **Road Graph Updates** → `4_road_adder.ipynb <examples/4_road_adder.ipynb>`_
   - Updates transport graphs with new roads and edges.
   - Analyzes connectivity improvement after graph modifications.

.. use-end

Data
----

Before running the examples, one can use the data from `tests
section <#tests/data>`__
and place it in the ``examples/data`` directory. You can use your own
data, but it must follow the structure described in the
`API documentation <https://aimclub.github.io/blocksnet/>`__.

Documentation
-------------

Detailed information and description of BlocksNet is available in
`documentation <https://blackcoster.github.io/transport_frames/>`__.

Project Structure
-----------------

The latest version of the library is available in the ``main`` branch.

The repository includes the following directories and modules:

-  `transport_frames <https://github.com/blackcoster/transport_frames/tree/main/transport_frames>`__
   - directory with the library code:

   -  graph - graph creation module
   -  frame - creation of the frame and priority roads module
   -  criteria - module for grading territory based on frame and criteria calculation
   -  indicators - module for area and territory indicators calculation
   -  road_adder - module for updating graph with new road edges
   -  utils - module containing utulity functions and consts

-  `tests <https://github.com/blackcoster/transport_frames/tree/main/tests>`__
   ``pytest`` testing
-  `examples <https://github.com/blackcoster/transport_frames/tree/main/examples>`__
   examples of how methods work
-  `docs <https://github.com/blackcoster/transport_frames/tree/main/docs>`__ -
   documentation sources

Developing
----------

.. developing-start

To start developing the library, one must perform following actions:

1. Clone the repository:
   ::

       $ git clone https://github.com/blackcoster/transport_frames

2. (Optional) Create a virtual environment as the library demands exact package versions:
   ::

       $ make venv

   Activate the virtual environment if you created one:
   ::

       $ source .venv/bin/activate

3. Install the library in editable mode with development dependencies:
   ::

       $ make install-dev

4. Install pre-commit hooks:
   ::

       $ pre-commit install

5. Create a new branch based on ``develop``:
   ::

       $ git checkout -b develop <new_branch_name>

6. Start making changes on your newly created branch, remembering to
   never work on the ``main`` branch! Work on this copy on your
   computer using Git to do the version control.

.. 7. Update
..    `tests <https://github.com/blackcoster/transport_frames/tree/main/tests>`__
..    according to your changes and run the following command:

..    ::

..          $ make test

..    Make sure that all tests pass.

.. 8. Update the
..    `documentation <https://github.com/blackcoster/transport_frames/tree/main/docs>`__
..    and **README** according to your changes.

.. 9.  When you're done editing and local testing, run:

..    ::

..          $ git add modified_files
..          $ git commit

..    to record your changes in Git, then push them to GitHub with:

..    ::

..             $ git push -u origin my-contribution

..    Finally, go to the web page of your fork of the transport_frames repo, and click
..    'Pull Request' (PR) to send your changes to the maintainers for review.

.. developing-end


License
-------

The project has `BSD-3-Clause license <./LICENSE>`__

Contacts
--------

.. contacts-start

You can contact us:


-  `IDU <https://idu.itmo.ru/en/contacts/contacts.htm>`__ - Institute of
   Design and Urban Studies
-  `Polina Krupenina <https://t.me/ratyear>`__ - project manager
-  `Alexander Morozov <https://t.me/insert_later>`__ - lead software engineer

.. contacts-end


.. |Documentation Status| image:: https://github.com/blackcoster/transport_frames/actions/workflows/documentation.yml/badge.svg?branch=main
   :target: https://blackcoster.github.io/transport_frames/
.. |PythonVersion| image:: https://img.shields.io/badge/python-3.10-blue
   :target: https://pypi.org/project/blocksnet/
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |Readme_ru| image:: https://img.shields.io/badge/lang-ru-yellow.svg
   :target: README-RU.rst