mkdir output/m46
th main.lua -regimes regimes/m46/regimes.lua -nDonkeys 10 -nGPU 4 -GPU 1 -seed 12891 -testList validation/val01.list -regression -model models/model_46.lua -batchSize 64  -save output/m46 -meta meta/meta.tsv -data data/train_1024 -momentum 0.90 -weightDecay 0 -sb 2  | tee output/m46/train_stdout.out
