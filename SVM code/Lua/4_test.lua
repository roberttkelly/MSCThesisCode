require 'torch'
require 'xlua'
require 'optim'
require 'image'

testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- test function
function test(dataset, model, config)

   config.epoch = config.epoch or 1
   model:evaluate()
   
   local nTTA = config.nTTA or 8

   local confusion = optim.ConfusionMatrix(dataset:getNumberClass())
   -- local vars
   local time = sys.clock()
   -- test over test data
   print('==> testing on test set:')
   local tMSE = 0
   local inputBuffer = torch.Tensor():cuda()
   local targetBuffer = torch.Tensor():cuda()
   local nTesting = dataset:getNumberTesting()
   local progressCnt = math.floor(nTesting/20)
   local outBuffer = torch.Tensor():cuda()

   for t = 1,nTesting do
      if t%progressCnt == 0 then 
         xlua.progress(t, nTesting) 
      end
      -- test sample
      local input, target = dataset:getTest(t, nTTA)
	  local batchMSE = 0
	  local outSum = 0
	  outBuffer:resize(target:size())

	  for s = 1,input:size(1),opt.batchSize do
	      sz = math.min(opt.batchSize, input:size(1)-s+1)
	      sz2 = input:size()
	      sz2[1] = sz
		  inputBuffer:resize(sz2)
		  inputBuffer:copy(input:narrow(1,s,sz))
	      tsz = target:size()
	      tsz[1] = sz
		  targetBuffer:resize(tsz)
		  targetBuffer:copy(target:narrow(1,s,sz))

		  local output = model:forward(inputBuffer)
		  local err = criterion:forward(output, targetBuffer)
	  
		  batchMSE = batchMSE + err*sz
		  --outSum = outSum + output:sum()
		  outBuffer:narrow(1,s,sz):copy(output)
	   end
	   tMSE = tMSE + batchMSE/input:size(1)

	--DRD specific code
	   local out = outBuffer:float()
	   --if config.AVG then
	   if config.L_R then
		   local avgOut = out:mean(1):round():clamp(1,5):squeeze()
		   confusion:add(avgOut[1], target[1][1])
		   confusion:add(avgOut[2], target[1][2])
	   elseif config.LR2 then
		   local avgOut = out:view(2, out:size(1)/2):mean(2):round():clamp(1,5):squeeze()
		   confusion:add(avgOut[1], target[1])
		   confusion:add(avgOut[2], target[target:size(1)/2+1])
	   else
		   local avgOut = torch.round(out:mean())
		   avgOut = math.max(math.min(avgOut, 5), 1)
		   confusion:add(avgOut, target[1])
	   end

	--[[
       else
		   local median
	       if (out:size(1) >= 2) then 
			   median = out:sort()[out:size(1)/2]
			   median = torch.round(median)
		   else
		       median = out[1]
		   end
		   median = math.max(math.min(median, 5), 1)
		   confusion:add(median, target[1])
	   end
	]]
   end
   local rMSE = math.sqrt(tMSE / nTesting)
	   
   -- timing
   time = sys.clock() - time
   time = time / nTesting
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   
   print('epoch: ' .. config.epoch .. ' + RMSE (test set) : ' .. rMSE )
   confusion:updateValids()
   print(confusion)
   local kappa = computeKappa(confusion.mat)
   print ("test kappa is " .. kappa)
   print ("")
   print ("")

   testLogger:add{['rMSE (test set)'] = rMSE .. "  "..kappa}

   -- save/log current net
   config.improved = false
   if (not config.bestError) or (config.bestError > rMSE) then
	   config.bestError = rMSE
	   config.last_improved_epoch = config.epoch
	   config.improved = true
	   saveModel(model, "err")
   end

   config.kappaImproved = false
   if (not config.bestKappa) or (config.bestKappa < kappa) then
	   config.bestKappa = kappa
	   config.kappaImproved = true
	   saveModel(model, "kappa")
   end
   --testError = rMSE
   config.testError = rMSE
   config.kappa = kappa

   collectgarbage()
end

function saveModel(model, _type)
   sanitize(model)
   print ("save model ")
   if (_type == "kappa") then
	   torch.save(paths.concat(opt.save, 'best_kappa_model.net'), model)
	   torch.save(paths.concat(opt.save, 'best_kappa_optimState_.t7'), optimState)
   else
	   torch.save(paths.concat(opt.save, 'best_model.net'), model)
	   torch.save(paths.concat(opt.save, 'best_optimState_.t7'), optimState)
   end
end
