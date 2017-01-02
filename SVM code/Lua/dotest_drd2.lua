require 'torch'
require 'nn'
require 'cudnn'
require 'nnx'
require 'paths'
require 'image'
require 'cutorch'
require 'csvigo'
paths.dofile('fbcunn_files/AbstractParallel.lua')
paths.dofile('fbcunn_files/ModelParallel.lua')
paths.dofile('fbcunn_files/DataParallel.lua')
paths.dofile('fbcunn_files/Optim.lua')
paths.dofile('util.lua')

torch.setdefaulttensortype('torch.FloatTensor')


cmd = torch.CmdLine()
cmd:text()
cmd:text('Test prediction script')
cmd:text()
cmd:text('Options:')
cmd:option('-gpuid', 1, 'gpu id')
cmd:option('-meta', "meta.tsv" , 'meta file')
cmd:option('-model', 'none', 'path to model')
cmd:option('-modelDef', 'model/model_vd3.lua', 'path to model')
cmd:option('-sb', 8, 'n sample per batch')
cmd:option('-data', 'data/test_images', 'root dir of test images')
cmd:option('-out', "test.out", 'fill output with ones.')
cmd:option('-regression', true, 'regression.')
cmd:option('-nTTA', 16, 'number of test augmentation')
cmd:option('-batchSize', 16, 'number of samples per Batch')
cmd:text()

opt = cmd:parse(arg or {})

dofile (opt.modelDef)
dofile('DataSet.lua')

opt.jitterFunc = nil

--opt.normalizer = normalizeImg
opt.normalizer = nil
opt.sampleSize = frameSize

local dataset = DataSet(opt, {"image", "level"})

-- load model
require 'cutorch'
require 'cunn'
cutorch.setDevice(opt.gpuid)

local nTTA = opt.nTTA or 32

model = torch.load(opt.model)
if opt.rmodel  then
	local p1,gp1 = model:getParameters()
	model = createModel()
	local p,gp = model:getParameters()
	p:copy(p1)
end

model:cuda()
model:evaluate()

index = 1
batchSize = opt.batchSize or opt.sb*opt.nTTA

t = 0

local outfp = io.open(opt.out , "w")
local fileList = {}

for f in io.lines(opt.data) do
	table.insert(fileList, f)
end

print "===> begin testing"

local inputs= torch.Tensor()
local outputs= torch.Tensor()
local inputBuffer = torch.Tensor():cuda()
local nSamplePerBatch = opt.sb or 4
local names = {}
local bag1 = {}

function predict(names, bag1)
	local sz = 0
	for _,item in ipairs(bag1) do
		sz = sz + item:size(1)
	end

	local sz1 = bag1[1]:size()
	sz1[1] = sz
	inputs:resize(sz1)
	outputs:resize(sz)

	local start = 1
	for _,item in ipairs(bag1) do
		inputs:narrow(1, start, item:size(1)):copy(item)
		start = start + item:size(1)
	end

	for i =1,inputs:size(1),batchSize do
		local sz = math.min(batchSize, inputs:size(1)-i+1)
	    local inputBatch = inputs:narrow(1, i, sz)
	    inputBuffer:resize(inputBatch:size())
	    inputBuffer:copy(inputBatch)
	    local out1 = model:forward(inputBuffer) -- fprop through model
		--outputs[{{i,i+sz-1}}] = out1
		outputs:narrow(1, i, sz):copy(out1)
	end

	if opt.LogSoftMax then
		outputs:exp()
	end

	outputs:resize(#bag1, bag1[1]:size(1))
	local pred = outputs:mean(2):resize(#bag1)

	start = 1
	for j,item in ipairs(bag1) do
		outfp:write(names[j]) 
		outfp:write(","..string.format("%0.4g", pred[j]) )
		outfp:write("\n")
	end

end

local tickSize = math.floor(#fileList/100)
for _,f in pairs(fileList) do
   -- for each image in dir,
    local basename = paths.basename(f)
    --local f = paths.concat(opt.data, f)
    if paths.filep(f) and string.sub(f, #f-4) == '.jpeg' then
		if ( index % tickSize ==0 ) then xlua.progress(index, #fileList) end
		index = index + 1;
		local im = dataset:getTestFromFile(f, nTTA)
		local tmp
		table.insert(bag1, im)
		table.insert(names, basename)

		if (#bag1 >= nSamplePerBatch) then
			predict (names, bag1, bag2)
			names={}
			bag1={}
		end
		if (index % 5 == 0 ) then
			collectgarbage('collect')
		end
   end
end

if (#bag1 > 0) then
	predict (names, bag1)
end
outfp:close()

