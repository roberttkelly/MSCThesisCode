--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
require 'xlua'

--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    nesterov = true,
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

local optimator = nn.Optim(model, optimState)

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local regimes = {
          --nEpoch,    LR,   WD,
        --{ 2, 1e-3,  1e-4 },
        { 30, 3e-3,  0 },
        { 30,  1e-3,  0 },
        { 1000,  1e-4,  0 }
    }

print( "training regims")
print( regimes)

local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end

	--so regimes can be changed while training
	if opt.regimes and opt.regimes ~= "none" then
		regimes = dofile(opt.regimes)
	end

    local cnt = 0
    for _, row in ipairs(regimes) do
	    cnt = cnt + row[1]
        if epoch <= cnt then
            return { learningRate=row[2], weightDecay=row[3] }, epoch == cnt-row[1]+1
        end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, loss_epoch
local tError
local confusion
local cnt = 0
local tickSize = math.floor(opt.epochSize/20)

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train(dataset)
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   confusion = optim.ConfusionMatrix(dataset:getNumberClass())

   confusion:zero()
   tError = 0
   cnt = 0

   local params, newRegime = paramsForEpoch(epoch)
	print (params)
   optimator:setParameters(params)
   if newRegime then
       -- Zero the momentum vector by throwing away previous state.
       optimator = nn.Optim(model, optimState)
   end
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()
   --model:cuda() -- get it back on the right GPUs.

   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0
   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
		 function(tid)
		 	local tid = tid or __threadid
            local inputs, labels = dataset:getBatch(tid)
			collectgarbage()
            return sendTensor(inputs), sendTensor(labels),tid
         end,
         -- the end callback (runs in the main thread)
		 function (inputs, labels, threadid)
			 trainBatch(inputs, labels)
			 dataset:refillId(threadid)
		 end
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize

   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t',
                       epoch, tm:time().real, loss_epoch, top1_epoch))
   print('\n')

   sanitize(model)
   collectgarbage()

  confusion:updateValids()
   print(confusion)
   local kappa = computeKappa(confusion.mat)
   print ("train kappa is " .. kappa)
   print('')
   print('')
   local avgError = math.sqrt(tError/opt.epochSize/opt.batchSize)
   print (avgError)
   config.trainError = avgError
   local msg = {['Error (train set)'] = avgError .. " "..params.learningRate.." "..kappa}
   print (msg)
   trainLogger:add(msg)
   --print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
          --epoch, batchNumber, opt.epochSize, timer:time().real, err, 
          --optimState.learningRate, dataLoadingTime))

end -- of train()
-------------------------------------------------------------------------------------------
-- create tensor buffers in main thread and deallocate their storages.
-- the thread loaders will push their storages to these buffers when done loading
local inputsCPU = torch.FloatTensor()
local labelsCPU = torch.FloatTensor()

-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()


local timer = torch.Timer()
local dataTimer = torch.Timer()
-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsThread, labelsThread)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()
   -- set the data and labels to the main thread tensor buffers (free any existing storage)
   receiveTensor(inputsThread, inputsCPU)
   receiveTensor(labelsThread, labelsCPU)

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local err, outputs = optimator:optimize(
       optim.sgd,
       inputs,
       labels,
       criterion)

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   tError = tError + err*inputs:size(1)
   recordConfusion(confusion, outputs, labelsCPU)

   -- Calculate top-1 error, and print information
	 if cnt%tickSize == 1 then 
	      xlua.progress(cnt, opt.epochSize) 
      end
	  cnt = cnt +1
	  collectgarbage()  

   dataTimer:reset()
end
