--uses step layer to retune m51
require 'nn'
require 'cudnn'
require 'rfgcunnx'
require '../JitterLayer'
require '../StepLayer'
require '../DoubleThrCriterion'
local img_jitter = require '../ImgJitter'
require '../SwitchWrapper'

frameSize = {3, 1024, 1024}
sampleSize = {3, 895, 895}

local saved_model = torch.load("output/m51_no_bg/m51_single.net")

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
	return img_jitter.getJDTransform2(srcWidth, srcHeight, outWidth, outHeight, 0.8, 0.95, false)
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
	  local model = nn.Sequential()
	  local features = {3, 16, 32, 64, 96, 128, 256, 384, 512}
	  local totalFeatures = 512*9

	  local fLayers = nn.Sequential()

	  fLayers:add(nn.JitterLayer(sampleSize[2], sampleSize[3], jitterFunc))
	  fLayers:add(nn.WarpAffine(sampleSize[2], sampleSize[3]))
	  fLayers:add(nn.GCN())

	  for i = 1,#features-1 do 
		    fLayers:add(makeConvLayer(features[i], features[i+1], 3, 1, 1))
			fLayers:add(nn.VeryLeakyReLU(0,0.1))
			fLayers:add(makeConvLayer(features[i+1], features[i + 1], 1, 1, 0))
			if i==#features-1 then
				fLayers:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
			else
				fLayers:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
			end
			fLayers:add(nn.VeryLeakyReLU(0,0.1))
	  end

	  model:add(fLayers)

	  model:add(nn.Dropout(0.5))
	  model:add(nn.Reshape(totalFeatures))
	  model:add(nn.Linear(totalFeatures, 1024))
	  model:add(nn.VeryLeakyReLU(0,0.1))
	  model:add(nn.Linear(1024, 1024))
	  model:add(nn.VeryLeakyReLU(0,0.1))
	  local ll = nn.Linear(1024, 1)
	  ll.bias:fill(3)
	  model:add(ll)

	  copyModule(saved_model.modules[1], model.modules[1])
	  copyModule(saved_model, model)

      local thr = {1.5, 2.5, 3.5, 4.5}
	  local alpha = {5, 5, 5, 5}
      model:add(nn.StepLayer(thr, alpha))

	  local dp  = model
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
