--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local defaultDir = 'output'

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-save',
               defaultDir ..'/imagenet_runs_oss',
               'subdirectory in which to save/log experiments')
    cmd:option('-data',
               'imagenet_raw_images/256',
               'Home of image dir')
    cmd:option('-seed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        2, 'number of donkeys to initialize (data loading threads)')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         250,    'Number of total epochs to run')
    cmd:option('-epochSize',       "none", 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       128,   'mini-batch size (1 = pure stochastic)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    ---------- Model options ----------------------------------
    cmd:option('-model',     'none', 'model file')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
	cmd:option('-regression',     false,       'treat the label as different classes')
	cmd:option('-testList',     "none",       'provide path to model to retrain with')
	cmd:option('-regimes',     "none",       'provide path to the regime')
	cmd:option('-meta',     "meta.tsv",       'provide path to model to retrain with')
	cmd:option('-mu',     false,       'whehter to update momentum')
	cmd:option('-finalMomentum',     0.999,       'final momentum')
	cmd:option('-sb',     4,       '# of samples merged in one batch in testing phase')

    cmd:text()

    local opt = cmd:parse(arg or {})
	opt.manualSeed = opt.seed
	opt.cache = opt.save
	opt.nThread = opt.nDonkeys
    return opt
end

return M
