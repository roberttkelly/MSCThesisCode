require 'nn'
require 'cudnn'
require 'rfgcunnx'
require '../JitterLayer'
require '../DoubleThrCriterion'
local img_jitter = require '../ImgJitter'
require '../SwitchWrapper'

frameSize = {3, 1024, 1024}
sampleSize = {3, 767, 767}

local saved_m46 = torch.load("output/m46/m46_single.net")

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
	  local features = {3, 16, 32, 64, 128, 256, 384, 512, 512}
	  local totalFeatures = 512*4

	  local model = nn.Sequential()

	  model:add(nn.JitterLayer(sampleSize[2], sampleSize[3], jitterFunc))
	  model:add(nn.WarpAffine(sampleSize[2], sampleSize[3]))
	  model:add(nn.GCN())

	  for i = 1,#features-1 do 
		    model:add(makeConvLayer(features[i], features[i+1], 3, 1, 1))
			model:add(nn.VeryLeakyReLU(0,0.1))
			model:add(makeConvLayer(features[i+1], features[i + 1], 1, 1, 0))
			model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
			model:add(nn.VeryLeakyReLU(0,0.1))
	  end
	  model:add(nn.Dropout(0.5))
	  model:add(nn.Reshape(totalFeatures))
	  model:add(nn.Linear(totalFeatures, 1024))
	  model:add(nn.VeryLeakyReLU(0,0.1))
	  model:add(nn.Linear(1024, 1024))
	  model:add(nn.VeryLeakyReLU(0,0.1))
	  model:add(nn.Linear(1024, 1))
	  copyModule(saved_m46, model)
	  model:add(nn.FourSigmoid())
	  --saved_m46 = nil

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
