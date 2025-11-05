# WORKFLOW — Tuning Sherpa with Apprentice

This directory contains files for performing Sherpa tuning with Apprentice on a high-performance computing (HPC) system, using the app-tools scripts. The setup is optimized for clusters using Condor as the job submission and scheduling system.

In `run_sherpa.sh`, the path to the Sherpa installation needs to be adjusted (line 30), and in `sherpa.jdf`, the path to the `run_sherpa.sh` file needs to be specified (line 1). In `run_app-build.sh`, the paths to the Rivet (line 6) and Apprentice installation (line 33) need to be adjusted, and in `build.jdf`, the path to the `run_app-build.sh` file need to be specified (line 1).

FOR A SINGLE RUNCARD/PROCESS

1. Prepare input files:
  - Create a parameter.json file with parameter ranges
  - Create template of Sherpa runcard

2. Create the grid:
  - Run: 'app-sample parameter.json N template.yaml'
  (alternatively use: 'app-tools-create_grid parameter.json template.yaml N')

3. Create an Initial-run directory and initialize Sherpa:
  - Run: 'mkdir Initial-run && cd Initial-run'
  - Run: 'Sherpa -I runcard.yaml' and 'Sherpa -e 0 runcard.yaml'
  (initializes processes and computes ME)

4. Prepare job submission (condor):
  - Run: 'app-tools-prepare_run_directories newscan n'
  (this generates n subfolders for each folder in newscan)
  - Run: 'mkdir ErrLogOut' (directory for the HPC log, error and output files of each job)
  - In the directory that contains newscan, run: 'condor_submit sherpa.jdf' (this submits N * n jobs on the HPC)

5. Combine .yoda files:
  - Run: 'app-tools-yodamerge newscan'
  (this combines the n .yoda files into one .yoda file using yodamerge)

6. Prepare reference data:
  - Copy the refrence data .yoda for each analysis from Rivet into a folder
  - Run: 'app-datadirtojson REFRENCE_DATA_FOLDER -o reference_data.json'

7. Prepare weights file:
  - Run: 'app-tools-write_weights YODA_FILE -o weights.txt'
  (you can use any .yoda file in newscan for this)

8. Proceed with apprentice tuning workflow:
  - Run: 'app-build newscan --order X,Y -w weights.txt -o app.json'
  - Run: 'app-tune2 weights.txt reference_data.json app.json [-s SURVEY_SIZE -r RESTARTS]'


FOR MULTIPLE RUNCARDS/PROCESSES

1. Prepare input files:
  - Create a parameter.json file with parameter ranges
  - Create templates of all Sherpa runcards
  - for each runcard you should use a separate directory

2. Create the grid:
  - Run: 'app-tools-create_grid parameter.json template_1.yaml N' for one of the runcards
  - Run: 'app-tools-create_grid PATH/TO/newscan_1 template_X.yaml N' for all other RUNCARDS (in their respective directory)
  (this will copy the grid created for the first runcard)

3. Do steps 3, 4 & 5 in the SINGLE RUNCARD guide above for each process (i.e. in their respective directory)

4. Combine the grids:
  - Run: 'mkdir -p merged/newscan'
  - Run: 'app-tools-yodamerge_directories PROCESS_1/newscan PROCESS_2/newscan [PROCESS_3/newscan ...] merged/newscan'
  (this will combine all the grids into a single newscan directory, keeping the params.dat files)

5. Proceed with steps 6, 7 & 8 in the SINGLE RUNCARD guide above in the merged directory (you can also scale the weights from each process by using 'app-tools-combine_weights PROCESS_1/weights.txt factor_1 PROCESS_2/weights.txt factor_2 [PROCESS_3/weights.txt factor_3 ...] -o merged/weights.txt')


BUILD THE SURROGATES ON THE HPC

1. Prepare weight files:
  - Run: 'app-tools-split_build_process weights.txt n'
  (this splits the weights file into n weight files of equal size, saved in the weight_files directory)

2. Prepare job submission (condor):
  - Change X,Y in build.jdf to the wanted polynomial orders
  - In the directory that contains newscan and weight_files, run: 'condor_submit build.jdf'
  
3. Combine surrogates:
  - Run: 'app-tools-split_build_process app_X_Y n'
  (this will combine the n partial surrogates into a single .json file)
