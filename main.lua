--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
paths.dofile('fbcunn_files/AbstractParallel.lua')
paths.dofile('fbcunn_files/ModelParallel.lua')
paths.dofile('fbcunn_files/DataParallel.lua')
paths.dofile('fbcunn_files/Optim.lua')
paths.dofile('DataSet.lua')
paths.dofile('ThreadDataSet.lua')

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)
print(opt)

torch.setdefaulttensortype('torch.FloatTensor')

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('util.lua')
paths.dofile(opt.model)
local dataset = paths.dofile('data.lua')
paths.dofile('model.lua')
paths.dofile('train.lua')
paths.dofile('4_test.lua')

epoch = opt.epochNumber
model:cuda()

local i = 1
while i <= opt.nEpochs do
   train(dataset)
   collectgarbage()
   if i % 5 == 0 then
	   test(dataset, model, opt)
	   collectgarbage()
   end

   epoch = epoch + 1

   if opt.mu then
      optimState.momentum = math.max(opt.momentum, math.min(1- 1/(epoch*2+1), opt.finalMomentum))
   end
   i = i+1

end
