--stip out the bloated saved models
require 'nn'
require 'cunn'
dofile ('util.lua')
require 'JitterLayer'
require 'rfgcunnx'
require 'StepLayer'
require 'BlurLayer'
require 'SwitchWrapper'
require 'VerticalDropout'
require 'cudnn'
paths.dofile('fbcunn_files/AbstractParallel.lua')
paths.dofile('fbcunn_files/ModelParallel.lua')
paths.dofile('fbcunn_files/DataParallel.lua')
paths.dofile('fbcunn_files/Optim.lua')

model = torch.load(arg[1]):float()
model1 = model.modules[1]:float()
torch.save(arg[2], model1)
