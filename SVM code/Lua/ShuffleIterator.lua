local ShuffleIterator = torch.class('ShuffleIterator')

function ShuffleIterator:__init(n)
	self.n = n
	self.i = 1
	self.randInd = torch.randperm(n)
end

function ShuffleIterator:nextInd()
	if (self.i > self.n) then
		self.randInd = torch.randperm(self.n)
		self.i = 1
	end
	local ind = self.randInd[self.i]
	self.i = self.i + 1
	return ind
end
