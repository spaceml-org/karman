.. _tutorials:

Tutorials
===============

Data Inputs Download and Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To download and pre-process the data, we provide a set of scripts, under `karman/scripts/input_data_prep`.

These are divided into three categories: data downloading, data pre-processing and data merging.

Note that these procedures can take quite some RAM & CPU time.

* Data downloading:
  for this, you have to go in the `scripts/input_data_prep` and run the following commands (note, if you want to download the data for specific dates, or onto specific directories, you can pass the arguments, e.g. --output_dir path/to/dir, and similar, as preferred):
  ```bash
  python download_tudelft_thermo.py
  python download_sw_indices.py
  python download_omniweb.py
  python download_goes.py
  python download_soho.py
  ```
  We also run the NRLMSISE-00 model and collect the output for both the nowcasting and forecasting model (note that
  if you want to change input/output path to directories, or configurations of the parallel processing, you can do so by specifying the arguments):
  ```bash
  python run_nrlmsise00.py
  python run_nrlmsise00_time_series.py
  ```
Once this is done, you can start the pre-processing.

* Data pre-processing:
  for this, you have to go in the `scripts/input_data_prep` and run the following commands (note, if you want to pre-process the data onto specific directories, e.g. --output_dir path/to/dir 
  you can do so by specifying the argument):
  ```bash
  python process_tudelft_data.py
  python process_sw_indices.py
  python process_omniweb_data.py
  python process_goes_data.py
  python process_soho_data.py
  ```
  Finally, you can merge the data.

  ```bash
  python omni_data.py
  python merge_sw_and_tudelft_data.py
  ```

* Data merging:
  for this, you have to go in the `scripts/input_data_prep` and run the following commands (note, if you want to merge the data onto specific directories, e.g. --output_dir path/to/dir
  you can do so by specifying the argument):

  ```bash
  python merge_data.py
  ```

Basics
^^^^^^^^^^^
These tutorials include some basic examples on how to use karman for simple tasks.

.. toctree::
  :maxdepth: 1
  
  notebooks/data_analysis.ipynb

Advanced
^^^^^^^^^^^
These tutorials are more advanced examples on how to leverage karman framework 
for more complex tasks (e.g. nowcasting, forecasting, etc.)

.. toctree::
  :maxdepth: 1

  notebooks/tutorial_nowcast.ipynb
  notebooks/tutorial_forecast.ipynb
