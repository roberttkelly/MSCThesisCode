require 'ShuffleIterator'

local BalancedIterator = torch.class('BalancedIterator')

function BalancedIterator:__init(classId, weight)
	self:split(classId)
	if weight then
	   assert (#weight == #self.classes, "# of weights must equal # of classes")
	   self.weight = torch.Tensor(weight)
	   self.weightSum = self.weight:sum()
	end
	self.i = 1
end

function BalancedIterator:setWeight(weight)
	self.weight = torch.Tensor(weight)
	self.weightSum = self.weight:sum()
end

function BalancedIterator:weightedPick()
    local r = torch.uniform(0, self.weightSum)
	local s = 0.0
	local w = self.weight
	local i
	for i = 1, w:size(1) do
	      s = s + w[i]
		 if r < s then return i end
	end
	return w:size(1) 
end

function BalancedIterator:nextClass()
	if self.weight then
		local ci = self:weightedPick()
	    return self.classes[ci]
	end

	if (self.i > self.n) then
		self.randInd = torch.randperm(self.n)
		self.i = 1
	end
	return self.classes[self.randInd[self.i]]
end


function BalancedIterator:split(classId)
	self.classCnt = {}
	local sz
	if (type(classId) == "table") then
		sz = #classId
	else
		sz = classId:size(1) -- assuming 1d tensor
	end

	self.classSet = {}
	for i = 1,sz do 
		local id = classId[i]
		self.classSet[id] = self.classSet[id] or {}
		table.insert(self.classSet[id], i)
	end
	local cnt = 0
	self.iteratorSet = {}
	local classes = {}
	for l, cl in pairs(self.classSet) do
		self.iteratorSet[l] = ShuffleIterator(#cl)
		cnt = cnt + 1
		table.insert(classes, l)
	end
	self.n = cnt
	self.i = 1
	self.randInd = torch.randperm(cnt)
	self.classes = classes
end

function BalancedIterator:nextInd()
	local cl = self:nextClass() 
	local i1 = self.iteratorSet[cl]:nextInd()
	local i2 = self.classSet[cl][i1]
	self.i = self.i + 1
	return i2
end
