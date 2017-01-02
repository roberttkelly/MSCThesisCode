mkdir -p output/m51_no_bg
th main.lua -nDonkeys 10 -nGPU 4 -GPU 1 -seed 12891 -testList validation/val01_no_big_gap.list -regression -model models/model_51.lua -batchSize 56 -save output/m51_no_bg -meta meta/meta_no_big_gap.tsv -data data/train_1024 -momentum 0.90 -weightDecay 0 -sb 2  -regimes regimes/m51_no_bg/regimes.lua | tee output/m51_no_bg/train_stdout.out
