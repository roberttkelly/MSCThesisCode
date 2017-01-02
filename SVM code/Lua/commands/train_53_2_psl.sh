mkdir output/m53_2_psl
th main_pseudo1.lua -nDonkeys 10 -nGPU 4 -GPU 1 -seed 121 -testList validation/val01_psl.list -regression -model models/model_53_2.lua -batchSize 72  -save output/m53_2_psl -meta meta/ens-105-psl_all_meta2.tsv -data data -momentum 0.90 -weightDecay 0 -sb 2 -regimes regimes/m53_2_psl/regimes.lua | tee output/m53_2_psl/train_stdout.out
