mkdir -p output/m55_bg_small
th main.lua -nDonkeys 10 -nGPU 4 -GPU 1 -seed 33343 -testList validation/val01_bg_small.list -regression -model models/model_55.lua -batchSize 72 -save output/m55_bg_small -meta meta/meta_bg_small.tsv -data data/train_1024 -momentum 0.90 -weightDecay 0 -sb 2  -regimes regimes/m55_bg_small/regimes.lua | tee output/m55_bg_small/train_stdout.out
