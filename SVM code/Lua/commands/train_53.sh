mkdir -p output/m53
th main.lua -nDonkeys 10 -nGPU 4 -GPU 1 -seed 12891 -testList validation/val01.list -regression -model models/model_53.lua -batchSize 72 -save output/m53 -meta meta/meta.tsv -data data/train_1024 -momentum 0.90 -weightDecay 0 -sb 2  -regimes regimes/m53/regimes.lua | tee output/m53/train_stdout.out
