require 'nn'
require 'cunn'
require 'rfgcunnx'
require 'cudnn'
require '../JitterLayer'
require '../DoubleThrCriterion'
require '../SwitchWrapper'
local img_jitter = require '../ImgJitter'

frameSize = {3, 512, 512}
sampleSize = {3, 480, 480}

local saved_m28 = torch.load("output/m28/m28_best_model.net")

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

local model_28_l1 = function()
	  local model = nn.Sequential()
	  local features = {3, 8, 32, 64, 128, 256, 512}
	  local totalFeatures = 512*4

	  model:add(nn.JitterLayer(sampleSize[2], sampleSize[3], jitterFunc))
	  model:add(nn.WarpAffine(sampleSize[2], sampleSize[3]))
	  --model:add(nn.ContrastJitter(0.8, 1.2, false))
	  model:add(nn.GCN())

	  for i = 1,1 do 
		    model:add(makeConvLayer(features[i], features[i + 1], 3, 1, 1))
			model:add(nn.VeryLeakyReLU(0,0.1))
			model:add(makeConvLayer(features[i + 1], features[i + 1], 3, 1, 1))
			model:add(nn.VeryLeakyReLU(0,0.1))
			model:add(makeConvLayer(features[i + 1], features[i + 1], 3, 1, 1))
			model:add(nn.VeryLeakyReLU(0,0.1))
			model:add(cudnn.SpatialMaxPooling(4, 4, 4, 4))
	  end
	  return model
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

local copyModule = function(from, to)
	for i=1,#to.modules do
		local from_l = from.modules[i]
		local to_l = to.modules[i]
		copyParam(from_l, to_l)
	end
end

createModel = function(batchSize) 
  local batchSize = batchSize or opt.batchSize or 8
  local model = nn.Sequential()
  local features = {8, 32, 64, 128, 256, 256}
  local totalFeatures = 512*9
  local featureLayer = nn.Sequential()
  local m28_l1 = model_28_l1()
  copyModule(saved_m28, m28_l1)
  local switchWrapper = nn.SwitchWrapper(m28_l1, false)
  featureLayer:add(switchWrapper)
  model.switchWrapper = switchWrapper

  featureLayer:add(nn.CyclicSlice())
  featureLayer:add(makeConvLayer(8, 32, 3, 1, 1))
  featureLayer:add(nn.VeryLeakyReLU(0, 0.1))
  featureLayer:add(makeConvLayer(32, 16, 3, 1, 1))
  featureLayer:add(nn.VeryLeakyReLU(0, 0.1))
  featureLayer:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  featureLayer:add(nn.CyclicRoll())

  featureLayer:add(makeConvLayer(64, 64, 3, 1, 1))
  featureLayer:add(nn.VeryLeakyReLU(0, 0.1))
  featureLayer:add(makeConvLayer(64, 32, 3, 1, 1))
  featureLayer:add(nn.VeryLeakyReLU(0, 0.1))
  featureLayer:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  featureLayer:add(nn.CyclicRoll())

  featureLayer:add(makeConvLayer(128, 128, 3, 1, 1))
  featureLayer:add(nn.VeryLeakyReLU(0, 0.1))
  featureLayer:add(makeConvLayer(128, 128, 3, 1, 1))
  featureLayer:add(nn.VeryLeakyReLU(0, 0.1))
  featureLayer:add(makeConvLayer(128, 64, 3, 1, 1))
  featureLayer:add(nn.VeryLeakyReLU(0, 0.1))
  featureLayer:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  featureLayer:add(nn.CyclicRoll())

  featureLayer:add(makeConvLayer(256, 256, 3, 1, 1))
  featureLayer:add(nn.VeryLeakyReLU(0, 0.1))
  featureLayer:add(makeConvLayer(256, 256, 3, 1, 1))
  featureLayer:add(nn.VeryLeakyReLU(0, 0.1))
  featureLayer:add(makeConvLayer(256, 128, 3, 1, 1))
  featureLayer:add(nn.VeryLeakyReLU(0, 0.1))
  featureLayer:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  featureLayer:add(nn.CyclicRoll())

  featureLayer:add(makeConvLayer(512, 512, 3, 1, 1))
  featureLayer:add(nn.VeryLeakyReLU(0, 0.1))
  featureLayer:add(makeConvLayer(512, 512, 3, 1, 1))
  featureLayer:add(nn.VeryLeakyReLU(0, 0.1))
  featureLayer:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  featureLayer:add(nn.CyclicPool())

  model:add(featureLayer)
  model:add(nn.Dropout(0.5))
   
  model:add(nn.Reshape(totalFeatures))
  model:add(nn.Linear(totalFeatures, 1024))
  model:add(nn.VeryLeakyReLU(0, 0.1))
  model:add(nn.Linear(1024, 1024))
  model:add(nn.VeryLeakyReLU(0, 0.1))
  model:add(nn.Linear(1024, 1))

  local criterion = nn.DoubleThrCriterion(1, 5)
  return model,criterion
end
