--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local ffi = require 'ffi'
local Threads = require 'threads'

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------
do -- start K datathreads (donkeys)
   local debug = false
   if (not debug) and opt.nDonkeys > 0 then
      local options = opt -- make an upvalue to serialize over to donkey threads
      donkeys = Threads(
         opt.nDonkeys,
         function()
            require 'torch'
			require 'DataSet'
			require 'ThreadDataSet'
			torch.setdefaulttensortype('torch.FloatTensor')
         end,
         function(idx)
            opt = options -- pass to all donkeys via upvalue
            tid = idx
            local seed = opt.manualSeed + idx
            torch.manualSeed(seed)
            print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
         end
      );
   else -- single threaded data loading. useful for debugging
      donkeys = {}
	  donkeys.cnt = 1
      function donkeys:addjob(f1, f2) 
            local tid = donkeys.cnt % opt.nDonkeys +1
	        f2(f1(tid))
	        donkeys.cnt = donkeys.cnt +1
      end
      function donkeys:synchronize() end
   end
end

nClasses = nil

opt.sampleSize = frameSize

local dataset = DataSet(opt, {"image", "level"})

dataset.nClasses = 5
nClasses = dataset.nClasses
dataset = ThreadDataSet(dataset, opt.nThread, opt.batchSize)
if opt.epochSize == "none" then
	opt.epochSize = math.ceil(dataset:getNumberTraining()/opt.batchSize) or 1000*opt.batchSize
end
return dataset
