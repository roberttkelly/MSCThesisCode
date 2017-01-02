mkdir -p output/m51_1
th main.lua -nDonkeys 10 -nGPU 4 -GPU 1 -seed 3431 -testList validation/val01.list -regression -model models/model_51_1.lua -batchSize 56 -save output/m51_1 -meta meta/meta.tsv -data data/train_1024 -momentum 0.90 -weightDecay 0 -sb 2 -regimes regimes/m51_1/regimes.lua | tee output/m51_1/train_stdout.out
