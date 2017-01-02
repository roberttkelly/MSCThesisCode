#!/bin/sh
mkdir output/m42
th main.lua -regimes regimes/m42/regimes.lua -nDonkeys 10 -nGPU 4 -GPU 1 -seed 12891 -testList validation/val01.list -regression -model models/model_42.lua -batchSize 96  -save output/m42 -meta meta/meta.tsv -data data/train_512 -momentum 0.90 -weightDecay 0 -sb 2  | tee output/m42/train_stdout.out
