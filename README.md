## Code Description

### Installation and Dependencies
1. [Install Torch and dependencies](http://torch.ch/docs/getting-started.htm)
2. [Install Torch wrappers for cudnn](https://github.com/soumith/cudnn.torch)
3. Install dependencies for multi-gpu support using the command in the Requirements section at [soumith/imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch)
4. Install custom nn layers

   ```
   sudo luarocks make rocks/rfgcunnx
   ```

### Preprocessing Images
```
th scaleCrop image_file output_dir image_size
```

* `image_file` is a file containing the _full path_ for each of the images to be scale and cropped
* `output_dir` is the output directory of the images
* `image_size` is the size of the square output images. Our models used size 512 and 1024 input images

We put the scaled images `data/train_512` and `data/train_1024` for training data and `data/test_512` and `data/test_1024` for testing data

### Training Submission Models

The `commands` directory contains the scripts we used to generate our submissions models

```
sh commands/train_42.sh
```

```
Options:
  -save            directory to save/log experiments [output/imagenet_runs_oss]
  -data            directory with training images [imagenet_raw_images/256]
  -seed            RNG seed [2]
  -GPU             Default GPU [1]
  -nGPU            Number of GPUs to use [1]
  -nDonkeys        number of data loading threads to initialize [2]
  -nEpochs         Number of training epochs [250]
  -epochSize       Number of batches per epoch [none]
  -epochNumber     Manual epoch number (useful on restarts) [1]
  -batchSize       mini-batch size [128]
  -LR              learning rate; if set, overrides default LR/WD recipe [0]
  -momentum        momentum [0.9]
  -weightDecay     weight decay [0.0005]
  -model           model file [none]
  -retrain         provide path to model to retrain with [none]
  -optimState      provide path to an optimState to reload from [none]
  -regression      treat the label as different classes [false]
  -testList        provide path to model to retrain with [none]
  -regimes         provide path to the regime [none]
  -meta            provide path to model to retrain with [meta.tsv]
  -mu              flag for updating momentum [false]
  -finalMomentum   final momentum [0.999]
  -sb              # of samples merged in one batch in testing phase [4]
```


Two models are serialized to disk during training: `output/m<model_id>/best_model.net` contains the model with the best criterion score  and `output/m<model_id>/best_kappa_model.net` contains the model achieving the best kappa score.

`output/m<model_id>` contains the training log

### Predictions

When using multiple GPUs during training, the serialized model will contain multiple copies, one for each GPU. Use the following command to extract a single copy of the model:

```
th getOneModule.lua  multi_gpu_model single_model_to_be_saved
```

Submission models are found in `output/m<model_id>/m<model_id>_single.net`

A sample prediction script is provided in `commands/predict_41.sh`:

```
th dotest_drd2.lua -gpuid 1 -model output/m41/m41_single.net -modelDef models/model_41.lua  -sb 2 -nTTA 64 -meta meta/meta.tsv -out output/m41/val01_pred.out -data validation/val01_fullpath_512.list -batchSize 32
```

```
Options:
  -gpuid      gpu id [1]
  -meta       meta file [meta.tsv]
  -model      path to model [none]
  -modelDef   path to model [model/model_vd3.lua]
  -sb         n sample per batch [8]
  -data       root dir of test images [data/test_images]
  -out        fill output with ones. [test.out]
  -regression regression. [true]
  -nTTA       number of test augmentation [16]
  -batchSize  number of samples per Batch [16]
```

If running out of memory, reduce batchSize

### Pooling Left and Right Images, Ensembling

```
Rscript --vanilla ens-tiered3-stage1.r path/to/training/images
Rscript --vanilla ens-tiered3-stage2.r path/to/training/images
java ScanKappa kappascan.tsv
Rscript --vanilla ens-tiered3-test.r path/to/test/images
```

The `ScanKappa` Java program is used to find the best kappa cutoff. The input file should contain tab-separated predictions and targets (with no header row). Cutoffs are taken from stdout
