--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'optim'

--[[
   1. Create Model
   2. Create Criterion
   3. If preloading option is set, preload weights from existing models appropriately
   4. Convert model to CUDA
]]--

-- 1.1. Create Network
print('=> Creating model from file: models/ ' .. opt.model)
model,criterion = createModel(opt.nGPU) -- for the model creation code, check the models/ folder


print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

-- 3. If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model = torch.load(opt.retrain)
end

-- 4. Convert model to CUDA
print('==> Converting model to CUDA')
model = model:cuda()
criterion:cuda()

collectgarbage()
