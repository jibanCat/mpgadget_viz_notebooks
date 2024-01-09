#!/bin/bash
#SBATCH --partition=development
#SBATCH --job-name=test
#SBATCH --time=0:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1

# this script was used to work on the Frontera cluster

hostname
date

ibrun python vis_sims_mpi.py -f "/scratch1/01317/yyang440/cosmo_11p_sims/sims_for_visual/cosmo_11p_Box1000_Part750_0536/output/PART_067/1/" -l 1000000 -s 8. -o 'vis_Box1000_Part750_0536_redshift0_r5_ycut250.png' -i 4000 -r 5 -y 250000 -w .1333
date
