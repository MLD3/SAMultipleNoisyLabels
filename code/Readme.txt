Note: an extended version of the paper is included here

To preprocess the synthetic data:
    No extra steps are required as this dataset gets generated on the fly
    
To preprocess the mimic data:
    Get access to MIMIC-III and download it (https://physionet.org/content/mimiciii/1.4/)
    Run preprocess_mimic.py
    Run FIDDLE on the files generated from the 'get_feats' function, see command below:
        python run.py --data_fname feats_for_fiddle.csv --population_fname ids_for_fiddle.csv --output_dir [directory for FIDDLE output] --T 7 --dt 1
    FIDDLE is open source and can be found here (https://github.com/MLD3/FIDDLE)
    
To get the statistics in Table 1:
    Run get_data.py with line 378 commented in and line 379 commented out
    
To train models/run an experiment:
    Run run.py with the following arguments
        To get results for Figure 3: --dataset synth --experiment pcen
        To get results for Figure 4: --dataset synth --experiment offset
        To get results for Figure 5: --dataset synth --experiment neg
        To get results for Table 2: --dataset mimic --experiment baseline
        To get results for Figure 6: --dataset mimic --experiment size
    *The results get saved to the results folder

To generate the results from the plots and tables:
    Change 'exp_name' in line 229 of process_results.py
        To get Figure 3: exp_name = 'pcen'
        To get Figure 4: exp_name = 'offset'
        To get Figure 5: exp_name = 'neg'
        To get Table 2: exp_name = 'mimic'
        To get Figure 6: exp_name = 'size'
    *The plots folder is prepopulated with results from our latest run for convenience of reproducibility

Uses:
    Python 3.9.7
    Numpy 1.20.3
    Pandas 1.3.4
    Pytorch 1.10.1
    Matplotlib 3.7.2
