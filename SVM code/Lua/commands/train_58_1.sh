mkdir -p output/m58_1
th main.lua -nDonkeys 10 -nGPU 4 -GPU 1 -seed 32431 -testList validation/val01.list -regression -model models/model_58_1.lua -batchSize 72 -save output/m58_1 -meta meta/meta.tsv -data data/train_1024 -momentum 0.90 -weightDecay 0 -sb 2 -regimes regimes/m58_1/regimes.lua | tee output/m58_1/train_stdout.out
