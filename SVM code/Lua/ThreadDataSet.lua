local util = dofile('util.lua')

local ThreadDataSet = torch.class('ThreadDataSet')

function ThreadDataSet:__init(dataset, nThread, batchSize)
	self.dataset = dataset
	self.batchSize = batchSize
	self.nThread = nThread
	local idSet = {}
	for i=1,nThread do
		table.insert(idSet, self:getBatchIdList())
	end
	self.idCache = idSet
end

function ThreadDataSet:getBatchIdList()
	local result = {}
	local ds = self.dataset
	for i = 1, self.batchSize do
		table.insert(result, ds:nextI())
	end
	return result
end

function ThreadDataSet:getBatch(threadId)
	return self.dataset:getBatch(self.idCache[threadId])
end

function ThreadDataSet:refillId(threadId)
	self.idCache[threadId] = self:getBatchIdList()
end

function ThreadDataSet:getTest(i, nTTA)
	return self.dataset:getTest(i, nTTA)
end

function ThreadDataSet:getNumberTesting()
	return self.dataset:getNumberTesting()
end

function ThreadDataSet:getNumberClass()
	return self.dataset:getNumberClass()
end

function ThreadDataSet:getNumberTraining()
	return self.dataset:getNumberTraining()
end

function ThreadDataSet:setBalanced(value)
	return self.dataset:setBalanced(value)
end

function ThreadDataSet:getClassCount()
	return self.dataset:getClassCount()
end
