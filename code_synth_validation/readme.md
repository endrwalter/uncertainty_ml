Step 1 — Generate the data 

    python3 synthetic_generator.py
    # creates synthetic_data/ with 56 condition folders + manifest.csv

Step 2 — Generate configs and commands file (once)

    python3 2_generate_synthetic_configs.py \
        --manifest  ../data/synthetic_data/manifest.csv \
        --template  config_synthetic_template.ini \ 
        --code_path ../code_classification \ 
        --script    main_calibrate.py \ 
        --output    ../data/synthetic_data/synthetic_commands.txt

Step 3 — Submit the array

    sbatch sbatch_synthetic_array.sh
    
