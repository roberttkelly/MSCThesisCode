require 'nn'
require 'cudnn'
require 'rfgcunnx'
require '../JitterLayer'
require '../DoubleThrCriterion'
local img_jitter = require '../ImgJitter'

frameSize = {3,512, 512}
sampleSize = {3,385, 385}

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

createModel = function(nGPU)
	  local model = nn.Sequential()
	  local features = {3, 32, 64, 128, 256, 384, 512, 512}
	  local totalFeatures = 512*4

	  model:add(nn.JitterLayer(sampleSize[2], sampleSize[3], jitterFunc))
	  model:add(nn.WarpAffine(sampleSize[2], sampleSize[3]))
	  --model:add(nn.ContrastJitter(0.8, 1.2, false))
	  model:add(nn.GCN())

	  for i = 1,1 do 
		    model:add(makeConvLayer(features[i], features[i + 1], 4, 1, 2))
			model:add(nn.VeryLeakyReLU(0,0.1))
			model:add(makeConvLayer(features[i + 1], features[i + 1], 4, 1, 1))
			model:add(nn.VeryLeakyReLU(0,0.1))
			model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
	  end

	  for i = 2,#features -1 do 
		    model:add(makeConvLayer(features[i], features[i + 1], 4, 1, 2))
			model:add(nn.VeryLeakyReLU(0,0.1))
			model:add(makeConvLayer(features[i + 1], features[i + 1], 4, 1, 1))
			model:add(nn.VeryLeakyReLU(0,0.1))
			model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
	  end
	  --model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))

	  model:add(nn.Dropout(0.5))
	  --model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
	  model:add(nn.Reshape(totalFeatures))
	  model:add(nn.Linear(totalFeatures, 1024))
	  model:add(nn.VeryLeakyReLU(0,0.1))
	  model:add(nn.Linear(1024, 1024))
	  model:add(nn.VeryLeakyReLU(0,0.1))
	  local ll = nn.Linear(1024, 1)
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
