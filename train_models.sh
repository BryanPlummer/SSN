#!/bin/bash

export PYTHONUNBUFFERED="True"

NUM_GPUS=$1
DATASET=$2
SHARE=$3
TAG=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:6:$len}

SCHEDULE="60 120 160"
GAMMA="0.2 0.2 0.2"
DECAY=5e-4
EPOCHS=200
BATCH_SIZE=128
DEPTH=28
WIDTH=10

case ${DATASET} in
    cifar10)
	    python main.py data --dataset ${DATASET} \
	    --depth ${DEPTH} --wide ${WIDTH} \
	    --share_type ${SHARE} --cutout --job-id ${TAG} \
	    --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
	    --decay ${DECAY} --schedule ${SCHEDULE} \
	    --gammas ${GAMMA} --ngpu ${NUM_GPUS} \
	    ${EXTRA_ARGS}
	;;
    cifar100)
            python main.py data --dataset ${DATASET} \
            --depth ${DEPTH} --wide ${WIDTH} \
            --share_type ${SHARE} --cutout --job-id ${TAG} \
            --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
            --decay ${DECAY} --schedule ${SCHEDULE} \
            --gammas ${GAMMA} --ngpu ${NUM_GPUS} \
            ${EXTRA_ARGS}
        ;;
    imagenet)
            python main.py data --dataset ${DATASET} \
            --depth ${DEPTH} --wide ${WIDTH} \
            --share_type ${SHARE} --job-id ${TAG} \
            --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
            --decay ${DECAY} --schedule ${SCHEDULE} \
            --gammas ${GAMMA} --ngpu ${NUM_GPUS} \
            ${EXTRA_ARGS}
        ;;
    *)
        echo "No dataset given"
        exit
        ;;
esac

