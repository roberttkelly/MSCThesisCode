mkdir output/m28
th main.lua -regimes regimes/m28/regimes.lua -nDonkeys 10 -nGPU 4 -GPU 1 -seed 12891 -testList validation/val01.list -regression -model models/model_28.lua -batchSize 32 -save output/m28 -meta meta/meta.tsv -data data/train_512 -momentum 0.90 -weightDecay 0 -sb 2  | tee output/m28/train_stdout.out
