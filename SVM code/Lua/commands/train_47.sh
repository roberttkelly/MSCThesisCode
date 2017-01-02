mkdir output/m47
th main.lua -regimes regimes/m47/regimes.lua -nDonkeys 10 -nGPU 4 -GPU 1 -seed 12891 -testList validation/val01.list -regression -model models/model_47.lua -batchSize 48 -save output/m47 -meta meta/meta.tsv -data data/train_1024 -momentum 0.90 -weightDecay 0 -sb 2  | tee output/m47/train_stdout.out
