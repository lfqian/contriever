#!/bin/bash

NSPLIT=128 #Must be larger than the number of processes used during training
FILENAME=/home/lq62/contriever/pmc_full_paper/oa_comm_xml.PMC001xxxxxx.baseline.2023-06-18.tar.gz_finished.txt
INFILE=${FILENAME}
TOKENIZER=bert-base-uncased
#TOKENIZER=bert-base-multilingual-cased
SPLITDIR=./tmp-tokenization-${TOKENIZER}-$(basename ${FILENAME})/
OUTDIR=./encoded-data/${TOKENIZER}/$(basename ${FILENAME} | cut -f 1 -d '.')
NPROCESS=8
mkdir -p ${SPLITDIR}
echo ${INFILE}
split -a 3 -d -n l/${NSPLIT} ${INFILE} ${SPLITDIR}
pids=()
i=0
while [ $i -lt $NSPLIT ]; do
    num=$(printf "%03d\n" $i)
    FILE=${SPLITDIR}${num}
    #we used --normalize_text as an additional option for mContriever
    python3 preprocess.py --tokenizer ${TOKENIZER} --datapath ${FILE} --outdir ${OUTDIR} &
    pids+=($!)
    if [ $((i % NPROCESS)) -eq 0 ]; then
        for pid in ${pids[@]}; do
            wait $pid
        done
    fi
    i=$((i + 1))
done

for pid in ${pids[@]}; do
    wait $pid
done

echo ${SPLITDIR}

rm -r ${SPLITDIR}