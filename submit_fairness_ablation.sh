#!/bin/bash
# Submits the ablation job array, then a dependent aggregation job.
set -eo pipefail

cd /scratch1/aqliang/CSCI567-ML-Project
mkdir -p logs

# Submit the 16-task array (baseline + 15 combos)
ARRAY_JOB=$(sbatch --parsable run_fairness_ablation_array.sbatch)
echo "Submitted array job: $ARRAY_JOB"

# Submit the aggregation job, which waits for all array tasks to finish
AGG_JOB=$(sbatch --parsable --dependency=afterany:${ARRAY_JOB} run_fairness_aggregate.sbatch)
echo "Submitted aggregation job: $AGG_JOB (depends on $ARRAY_JOB)"
