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

  .. code-block:: console

      $ python download_tudelft_thermo.py
      $ python download_sw_indices.py
      $ python download_omniweb.py
      $ python download_goes.py
      $ python download_soho.py

Once this is done, you can start the pre-processing.

* Data pre-processing:
  for this, you have to go in the `scripts/input_data_prep` and run the following commands (note, if you want to pre-process the data onto specific directories, e.g. --output_dir path/to/dir
  you can do so by specifying the argument):

  .. code-block:: console

      $ python process_tudelft_data.py
      $ python process_sw_indices.py
      $ python process_omniweb_data.py
      $ python process_goes_data.py
      $ python process_soho_data.py

* NRLMSISE-00 data preparation
  We also run from `scripts/input_data_prep` the NRLMSISE-00 model and collect the output for both the nowcasting and forecasting model (note that
  if you want to change input/output path to directories, or configurations of the parallel processing, you can do so by specifying the arguments):

  .. code-block:: console

      $ python run_nrlmsise00.py
      $ python run_nrlmsise00_time_series.py


* Data merging:
  for this, you have to go in the `scripts/input_data_prep` and run the following commands (note, if you want to merge the data onto specific directories, e.g. --output_dir path/to/dir
  you can do so by specifying the argument):

  .. code-block:: console

      $ python merge_omni_data.py
      $ python merge_sw_and_tudelft_data.py

* Training nowcasting model
  Once you downloaded and prepared the data, you can either directly move to the tutorials,
  to learn how to use the models for inference, or if you are interested in training, then
  you can check the script under: `scripts/training/train_nowcasting_karman.py` to learn
  how to setup the training. Note that we have not open-sourced the training script for the
  forecasting models yet.

  An example usage is:

  .. code-block:: console

      $ python train_nowcasting_karman.py --batch_size 64 --lr 0.0001 --epochs 10 --hidden_layer_dim 48 --hidden_layers 3


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
