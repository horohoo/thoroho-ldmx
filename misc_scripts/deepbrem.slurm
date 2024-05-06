#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --output=log/deepbrem_%a.out
#SBATCH --partition=standard
#SBATCH -A ldmx

source "$LDMX_BASE/ldmx-sw/scripts/ldmx-env.sh"

ldmx fire /sfs/qumulo/qhome/tgh7hx/ldmx/thoroho-ldmx/misc_scripts/config_deepbrem.py ${SLURM_ARRAY_TASK_ID} /scratch/tgh7hx/deepbrem/deepbrem_${SLURM_ARRAY_TASK_ID}.root 
