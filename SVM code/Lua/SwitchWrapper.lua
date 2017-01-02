local SwitchWrapper,parent = torch.class('nn.SwitchWrapper', 'nn.Module')

function SwitchWrapper:__init(module, on)
	parent.__init(self)
	self.module = module
	self.on = on or false
	if not self.on then
		self.module:evaluate()
	end
	self.gradInput = self.module.gradInput
	self.output = self.module.output
end

function SwitchWrapper:switchOn()
	self.on = true
end

function SwitchWrapper:switchOff()
	self.on = false
	self.module:evaluate()
end

function SwitchWrapper:parameters()
    if self.on then
		return self.module:parameters()
	end
end

function SwitchWrapper:updateOutput(input)
    self.output =  self.module:updateOutput(input)
    return self.output
end

function SwitchWrapper:updateGradInput(input, gradOutput)
    if self.on then
		self.gradInput =  self.module:updateGradInput(input, gradOutput)
	end
    return self.gradInput
end

function SwitchWrapper:accGradParameters(input, gradOutput, scale)
    if self.on then
		return self.module:accGradParameters(input, gradOutput, scale)
	end
end

function SwitchWrapper:accUpdateGradParameters(input, gradOutput, lr)
    if self.on then
		return self.module:accUpdateGradParameters(input, gradOutput, lr)
	end
end

function SwitchWrapper:sharedAccUpdateGradParameters(input, gradOutput, lr)
    if self.on then
		return self.module:sharedAccUpdateGradParameters(input, gradOutput, lr)
	end
end

function SwitchWrapper:training()
	if self.on then
		self.module:training()
	end
end

function SwitchWrapper:evaluate()
	if self.on then
		self.module:evaluate()
	end
end

function SwitchWrapper:listModules()
	local modules = {self}
	local mds = self.module:listModules()
	for _,m in pairs(mds) do
		table.insert(modules, m)
	end
	return modules
end
function SwitchWrapper:__tostring__()
	  return torch.type(self)
end
