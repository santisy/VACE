BATCH_N=8
DATA_NAME=data_1105_small
MAX_PROMPTS=2

#for BATCH_ID in 8 9; do
for BATCH_ID in $(seq 0 $(( $BATCH_N - 1 ))); do
    RJOB_NAME=data-dd-batch${BATCH_ID}
    rjob delete ${RJOB_NAME}
    rjob submit \
        --name=${RJOB_NAME} \
        --gpu=1 \
        --memory=100000 \
        --cpu=12 \
        --charged-group=idc2_gpu \
        --private-machine=group \
        --mount=gpfs://gpfs1/yangdingdong:/mnt/shared-storage-user/yangdingdong \
        --image=registry.h.pjlab.org.cn/ailab-idc2-idc2_gpu/jianglihan:base -- \
        bash -exc "/mnt/shared-storage-user/yangdingdong/base_command.sh ${BATCH_N} ${BATCH_ID} ${DATA_NAME} ${MAX_PROMPTS}"

done
