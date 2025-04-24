#!/bin/bash

algorithms=("dgl" "f3s" "dfgnn_hyper" "dfgnn_tiling" "flashSparse")
datasets=("ZINC" "Peptides-struct" "Peptides-func" "PascalVOC-SP" "COCO-SP")

for alg in "${algorithms[@]}"
do
    for dataset in "${datasets[@]}"
    do
        echo "Running algorithm: $alg on dataset: $dataset"
        python eval.py --alg $alg --dataset $dataset
    done
done

