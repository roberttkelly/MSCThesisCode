require 'csvigo'
require('BalancedIterator')
require 'pl'
stringx.import()

local PSLDataSet,parent = torch.class('PSLDataSet', 'DataSet')

function PSLDataSet:__init(config, header)
	parent.__init(self, config, header)
	config.balanced = true
	local train_test_label = self:countTrainTest(self.trainId)
	self.trainIterator = BalancedIterator(train_test_label, config.weight)
end

function PSLDataSet:countTrainTest(idList)
	local cls = {}
	for _,str in pairs(idList) do
		if str:startswith("test") then
			table.insert(cls, 2)
		else
			table.insert(cls, 1)
		end
	end
	return cls
end
