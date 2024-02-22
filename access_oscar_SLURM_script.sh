#!/bin/bash
# --- this job will be run on any available node in the "gpu" partition
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="Slurm Simple Test Job"
#SBATCH --error="my_job.err"
#SBATCH --output="my_job.output"
# --- specify the partition (queue) name
#SBATCH --partition="gpu"
python3 access_oscar.py