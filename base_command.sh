!#/bin/bash

set -exo pipefail

BATCH_N=$1
BATCH_ID=$2
DATA_NAME=$3
MAX_PROMPTS=$4

__conda_setup="$('/mnt/shared-storage-user/yangdingdong/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then                                                           
    eval "$__conda_setup"                                                       
else                                                                            
    if [ -f "/mnt/shared-storage-user/yangdingdong/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/shared-storage-user/yangdingdong/miniconda3/etc/profile.d/conda.sh"
    else                                                                        
        export PATH="/mnt/shared-storage-user/yangdingdong/miniconda3/bin:$PATH"
    fi                                                                          
fi                                                                              
unset __conda_setup

conda activate wan

python batch_run.py \
    --input_root inputs/${DATA_NAME} \
    --output_root results/${DATA_NAME}_batch${BATCH_ID} \
    --batch_id ${BATCH_ID} \
    --batch_n ${BATCH_N} \
    --max_prompts_per_video ${MAX_PROMPTS} \
    --simplified \
    --reorg
