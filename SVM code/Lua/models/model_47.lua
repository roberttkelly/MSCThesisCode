require 'nn'
require 'cunn'
require 'rfgcunnx'
require 'cudnn'
require '../JitterLayer'
require '../DoubleThrCriterion'
local img_jitter = require '../ImgJitter'

frameSize = {3, 1024, 1024}
sampleSize = {3, 724, 724}

makeConvLayer = function(f1, f2, kw, dw, pw)
  local layer = cudnn.SpatialConvolution(f1, f2, kw, kw, dw, dw, pw, pw)
  local nl = kw * kw * f1
  local std = math.sqrt(2 / nl)
  local tmp = torch.randn(layer.weight:size()):mul(std)
  layer.weight:copy(tmp)
  layer.bias:zero()
  return layer
end

makeLinear = function(nIn, nOut)
  local layer = nn.Linear(nIn, nOut)
  local nl = nIn
  local std = math.sqrt(2 / nl)
  local tmp = torch.randn(layer.weight:size()):mul(std)
  layer.weight:copy(tmp)
  layer.bias:zero()
  return layer
end

local jitterFunc = function(srcWidth, srcHeight, outWidth, outHeight)
	return img_jitter.getJDTransform(srcWidth, srcHeight, outWidth, outHeight, 0.8, 0.95, false)
end

createModel = function(nGPU)
  local model = nn.Sequential()
  local features = {3, 16, 32, 32, 64, 64, 128, 128, 192, 192, 256, 256, 384, 384, 512, 512}
  --local features = {3, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16}
  local totalFeatures = 512 * 2 * 2

  model:add(nn.JitterLayer(sampleSize[2], sampleSize[3], jitterFunc))
  model:add(nn.WarpAffine(sampleSize[2], sampleSize[3]))
  model:add(nn.GCN())

  for i = 1, #features - 1 do 
    model:add(makeConvLayer(features[i], features[i+1], 3, 1, 1))
    model:add(nn.VeryLeakyReLU(0, 0.1))
    model:add(nn.FracSpatialMaxPooling(1.414, 0,true, true))
  end
   
  model:add(nn.Dropout())
  model:add(nn.Reshape(totalFeatures))
  model:add(makeLinear(totalFeatures, 1024))
  model:add(nn.VeryLeakyReLU(0, 0.1))
  model:add(makeLinear(1024, 1024))
  model:add(nn.VeryLeakyReLU(0, 0.1))
  local ll = makeLinear(1024, 1)
  ll.bias:fill(3)
  model:add(ll)

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

