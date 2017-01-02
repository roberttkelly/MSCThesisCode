require 'nn'
require 'cudnn'
require 'rfgcunnx'
require '../JitterLayer'
local img_jitter = require '../ImgJitter'

frameSize = {3,512, 512}
sampleSize = {3,480, 480}

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

createModel = function()
	  local model = nn.Sequential()
	  local features = {3, 8, 32, 64, 128, 256, 512}
	  local totalFeatures = 512*4

	  model:add(nn.JitterLayer(sampleSize[2], sampleSize[3], jitterFunc))
	  model:add(nn.WarpAffine(sampleSize[2], sampleSize[3]))
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

	  for i = 2,#features -1 do 
		    model:add(makeConvLayer(features[i], features[i + 1], 3, 1, 1))
			model:add(nn.VeryLeakyReLU(0,0.1))
			model:add(makeConvLayer(features[i + 1], features[i + 1], 3, 1, 1))
			model:add(nn.VeryLeakyReLU(0,0.1))
			model:add(makeConvLayer(features[i + 1], features[i + 1], 3, 1, 1))
			model:add(nn.VeryLeakyReLU(0,0.1))
			if i >= #features -3 then
				model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
			else
				model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
	 		end
	  end

	  model:add(nn.Dropout(0.5))
	  model:add(nn.Reshape(totalFeatures))
	  model:add(nn.Linear(totalFeatures, 1024))
	  model:add(nn.VeryLeakyReLU(0,0.1))
	  model:add(nn.Linear(1024, 1024))
	  model:add(nn.VeryLeakyReLU(0,0.1))
	  model:add(nn.Linear(1024, 1))
	  local criterion = nn.MSECriterion()
	  return model, criterion
end
