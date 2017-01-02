local DoubleThrCriterion, parent = torch.class('nn.DoubleThrCriterion', 'nn.Criterion')

function DoubleThrCriterion:__init(thr1, thr2)
    parent.__init(self)
    self.sizeAverage = true
    self.thr1 = thr1
    self.thr2 = thr2
    self.MSECr = nn.MSECriterion()
    self.filter1 = torch.Tensor()
    self.filter2 = torch.Tensor()
    self.tmp = torch.Tensor()
end

function DoubleThrCriterion:updateOutput(input, target)
	self.filter1:resizeAs(input)
	self.filter2:resizeAs(input)
	self.tmp:resizeAs(input)

	torch.lt(self.filter1, input, self.thr1):typeAs(input)
	torch.lt(self.filter2, target, self.thr1 + 1e-12):typeAs(input)
	self.filter1:cmul(self.filter2)
	self.tmp:fill(self.thr1):cmul(self.filter1)
	self.filter1:add(-1):mul(-1)
	input:cmul(self.filter1):add(self.tmp)

	torch.gt(self.filter1, input, self.thr2):typeAs(input)
	torch.gt(self.filter2, target, self.thr2 - 1e-12):typeAs(input)
	self.filter1:cmul(self.filter2)
	self.tmp:fill(self.thr2):cmul(self.filter1)
	self.filter1:add(-1):mul(-1)
	input:cmul(self.filter1):add(self.tmp)
    self.ouputput = self.MSECr:updateOutput(input, target)
	return self.ouputput
end

function DoubleThrCriterion:cuda()
	parent.cuda(self)
	self.MSECr:cuda()
end

function DoubleThrCriterion:updateGradInput(input, target)
    self.gradInput =  self.MSECr:updateGradInput(input, target)
	return self.gradInput
end
