## Queueing commands for the HPC

sbatch --time=10:00:00 --array=1-10 --export=input="cds3/<task>-inputs.txt",seed="--seed 0",sample="--sample 1" cds3/ensemble-run-gpu

python sampnn.py --ensemble 2 --evaluate --fold-id 3 --data-path data/datasets/expt-non-metals.csv --test-size 0.2 --seed --sample 48


# expt-inputs
This file contains the input hyperparameters for the experimental bandgaps

# mp-inputs
This file contains the input hyperparameters for the materials project bandgaps

# oqmd-inputs
This file contains the input hyperparameters for the oqmd formation enthalpies

# transfer-inputs
This file contains the input hyperparameters for the materials project to experimental bandgaps fine tuning task