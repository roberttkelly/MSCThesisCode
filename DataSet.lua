require 'csvigo'
require 'xlua'
require 'paths'
require 'nn'
require 'image'

require('BalancedIterator')
require('ShuffleIterator')
require('util')

local DataSet = torch.class('DataSet')

function DataSet:__init(config, header)
	self.config = config
	self.header = header
	self:loadMeta(config, header)

	if config.balanced then
		local t = self.trainTarget
		self.trainIterator = BalancedIterator(t, config.weight)
	else
		self.trainIterator = ShuffleIterator(self.nTraining)
	end
	self.jitter = config.jitterFunc
	self.normalizer = config.normalizer 

	if config.cacheImg then
		self.cache = {}
	end

	if config.sampleSize then
		self.sampleSize = config.sampleSize
	end
end

function DataSet:setBalanced(isBalanced, weight)
	if isBalanced then
self.trainIterator = BalancedIterator(self.trainTarget, weight)
	else
		self.trainIterator = ShuffleIterator(self.nTraining)
	end
end

function DataSet:setIteratorWeight(weight)
	if torch.type(self.trainIterator) == "BlancedIterator" then
		self.trainIterator:setIteratorWeight(weight)
	end
end

function DataSet:setTargetEncoder(encoder)
	self.targetEncoder = encoder
end

function DataSet:saveTestMeta(metafile, header, id2label)
	local fp = io.open(metafile, "w")
	fp:write(header[1].."\t"..header[2].. "\n")

	if id2label then 
		for i=1,self.nTesting do
		   fp:write(self.testId[i], "\t", self.testTarget[i], "\t", id2label[testTarget[i]], "\n")
		end
	else
		for i=1,self.nTesting do
		   fp:write(self.testId[i], "\t", self.testTarget[i], "\n")
		end
	end
    fp:close()
end

function DataSet:getNumberClass()
	return self.nClasses
end

function DataSet:getNumberTesting()
	return self.nTesting
end

function DataSet:getNumberTraining()
	return self.nTraining
end

function DataSet:getClassCount()
	return self.classCounts
end

function DataSet:tableToNumber(tbl)
	local result = {}
	for _,v in ipairs(tbl) do
	    table.insert(result, tonumber(v)+1)
	    --table.insert(result, (tonumber(v)+3)/10)
	end
	return result
end

function DataSet:countClass(target)
	local cnts = {}
	for i=1,#target do
	   local id = target[i]
	   cnts[id] = cnts[id] or 0
	   cnts[id] = cnts[id] + 1
	end
	self.classCounts = cnts
end

function DataSet:loadMeta(config, header)
	local header = header or {"ID", "VALUE"}
	local metaFile = config.meta or 'meta.tsv'
	local csvdata = csvigo.load{path=metaFile, verbose=false, separator="\t"}
	self.nSamples = #csvdata[header[1]]

	local sampleID = csvdata[header[1]]

	local target
	if config.regression then
		target = self:tableToNumber(csvdata[header[2]])
		if config.countClass then  --even for regression in some cases
			local  tmp = {}
			local cnt = 0
			for _,t in pairs (target) do
				if not tmp[t] then
					tmp[t] = 1
					cnt = cnt +1
				end
			end
			self.nClasses = cnt
		end
	else
		local classId, label2id, id2label = util.key2int(csvdata[header[2]])
		self.nClasses = #id2label
		target = classId
	    self:countClass(classId)
	end

	local pctTrain = config.pctTrain or 0.9

	if config.testList and config.testList ~= "none" then -- test list is provided
		local sample2id = {}
		for i,k in ipairs (sampleID) do
			sample2id[k] = target[i]
		end

		self.testId = {}
		for l in io.lines(config.testList) do
			table.insert(self.testId, l)
		end

		self.nTesting = #self.testId
		self.nTraining = self.nSamples - self.nTesting

		local testSet = {}
		local tmp = {}
		for i,l in ipairs(self.testId) do
			assert (sample2id[l], l .. " does not exist in the meta file")
			table.insert(tmp, sample2id[l])
			testSet[l] = 1
		end
		self.testTarget = tmp

		self.trainId = {}
		tmp = {}
		for i,l in ipairs(sampleID) do
			if not testSet[l] then
				table.insert(self.trainId, l)
				table.insert(tmp, target[i])
			end
		end
		self.trainTarget = tmp
	else
		local trIndices, tsIndices
		trIndices, tsIndices = randomSampling(self.nSamples, pctTrain)

		self.nTraining = trIndices:size(1)
		if tsIndices then
			self.nTesting = tsIndices:size(1)
		else
			self.nTesting = 0
		end

		self.trainId = {}
		self.testId = {}
		self.trainTarget = {}
		self.testTarget = {}

		for i=1,self.nTraining do
		   self.trainTarget[i] = target[trIndices[i]]
		   self.trainId[i] = sampleID[trIndices[i]]
		end

		for i=1,self.nTesting do
		   self.testTarget[i] = target[tsIndices[i]]
		   self.testId[i] = sampleID[tsIndices[i]]
		end

		if config.testMeta then
		   self:saveTestMeta(config.testMeta, header)
		end
	end

	collectgarbage()
	--=========
	print('Number of Samples: ' .. self.nSamples)
	print('Training samples: ' .. self.nTraining)
	print('Testing samples: ' .. self.nTesting)
end

function DataSet:loadImg (fileName)
	local im
	if self.cache then
		im = self.cache[fileName]
		if not im then
			im = image.load(fileName)
	        self.cache[fileName] = im
		end
	else
		im = image.load(fileName)
	end
	return im
end

function DataSet:getSampleFromFile(filename)
	local im = self:loadImg(filename)
	if self.jitter then
		im = self.jitter(im)
	end
	return im
end

function DataSet:nextI()
	return self.trainIterator:nextInd()
end

function DataSet:getTrainSample(i)
	local filename = paths.concat(self.config.data, self.trainId[i])
	local im= self:getSampleFromFile(filename)
	local target = self.trainTarget[i]
	return im, target
end

function DataSet:getBatch(n)
	local idSet = {}
	if type(n) == "table" then
		idSet = n
	else
		for i=1,n do
			table.insert(idSet, self:nextI())
		end
	end

	local img, gt
	local sampleSize = self.sampleSize or sampleSize -- otherwise from global variable
	img = torch.Tensor(#idSet, sampleSize[1], sampleSize[3], sampleSize[2])
	gt = {}
	for i=1,#idSet do
	    img[i], gt[i] = self:getTrainSample(idSet[i])
	end
	if (self.normalizer) then
		img = self.normalizer(img, self.config.byChannel)
	end
	if self.targetEncoder then
		gt = self.targetEncoder:encode(gt)
	end
	gt = torch.Tensor(gt)
	return img, gt
end

function DataSet:getTest(i, nTTA)
	local filename = paths.concat(self.config.data, self.testId[i])
	local im = self:getTestFromFile(filename, nTTA)
	local gt, gt1
	gt = self.testTarget[i]
	if self.targetEncoder then
		gt = self.targetEncoder:encode(gt)
	end
	gt1 = {}
	for i = 1,nTTA do
		table.insert(gt1, gt)
	end
	gt1 = torch.Tensor(gt1)
	return im, gt1
end

function DataSet:getTestFromFile(filename, nTTA)
	local im = self:loadImg(filename)
	local sampleSize = self.sampleSize or sampleSize -- otherwise from global variable
	local o = torch.Tensor(nTTA, sampleSize[1], sampleSize[3], sampleSize[2])

	for i=1,nTTA do
		if self.jitter then
		   o[i] = self.jitter(im)
		else
		   o[i] = im
		end
	end

	if self.normalizer then
		o = self.normalizer(o, self.config.byChannel)
	end
	return o
end
