--experimenting with the step layer
require 'nn'
require 'cudnn'
require 'rfgcunnx'
require '../JitterLayer'
require '../StepLayer'
require '../DoubleThrCriterion'
require '../SwitchWrapper'
local img_jitter = require '../ImgJitter'

frameSize = {3, 1024, 1024}
sampleSize = {3, 767, 767}


makeConvLayer = function(f1, f2, kw, dw, pw)
	  local layer = cudnn.SpatialConvolution(f1, f2, kw, kw, dw, dw, pw, pw)
	  local nl = kw * kw * f1
	  local std = math.sqrt(2 / nl)
	  local tmp = torch.randn(layer.weight:size()):mul(std)
	  layer.weight:copy(tmp)
	  layer.bias:zero()
	  return layer
end

local jitterFunc = function(srcWidth, srcHeight, outWidth, outHeight)
	return img_jitter.getJDTransform(srcWidth, srcHeight, outWidth, outHeight, 0.8, 0.95, false)
end

local copyParam = function (from, to)
	if from.weight then
		to.weight:copy(from.weight)
	end
	if from.bias then
		to.bias:copy(from.bias)
	end
	if from.gradBias then
		to.gradBias:copy(from.gradBias)
	end
	if from.gradWeight then
		to.gradWeight:copy(from.gradWeight)
	end
end

local copyModule2 = function(from, to)
	local layers_to_copy = {1,2,3}
	for i=1,8 do
		table.insert(layers_to_copy, 4+5*(i-1))
		table.insert(layers_to_copy, 6+5*(i-1))
    end
	for _,i in pairs(layers_to_copy) do
		local from_l = from.modules[i]
		local to_l = to.modules[i]
		copyParam(from_l, to_l)
	end
end

local copyModule = function(from, to)
	for i=1,#to.modules do
		local from_l = from.modules[i]
		local to_l = to.modules[i]
		copyParam(from_l, to_l)
	end
end

createModel = function(nGPU)
	  local features = {3, 16, 32, 64, 128, 256, 384, 512, 768}
	  local totalFeatures = 768*4

	  local model = nn.Sequential()
	  local lowerModel = nn.Sequential()
	  local saved_m58 = torch.load("output/m58/m58_single.net")
	  model:add(saved_m58.modules[1].module)
	  for i = 2, #saved_m58.modules do
		  model:add(saved_m58.modules[i])
      end
	  	
	  local dp = model
	  local nGPU = nGPU or 1
	  if nGPU > 1 then
		  assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
		  local model_single = model
		  dp = nn.DataParallel(1)

		  for i=1,nGPU do
			 cutorch.withDevice(i, function()
                  dp:add(model_single:clone())
			 end)
		  end
      end

	  local criterion = nn.DoubleThrCriterion(1, 5)
	  return dp, criterion
end
