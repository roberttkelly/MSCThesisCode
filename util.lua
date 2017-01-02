local ffi=require 'ffi'
------ Some FFI stuff used to pass storages between threads ------------------
ffi.cdef[[
void THFloatStorage_free(THFloatStorage *self);
void THLongStorage_free(THLongStorage *self);
]]

function setFloatStorage(tensor, storage_p)
   assert(storage_p and storage_p ~= 0, "FloatStorage is NULL pointer");
   local cstorage = ffi.cast('THFloatStorage*', torch.pointer(tensor:storage()))
   if cstorage ~= nil then
      ffi.C['THFloatStorage_free'](cstorage)
   end
   local storage = ffi.cast('THFloatStorage*', storage_p)
   tensor:cdata().storage = storage
end

function setLongStorage(tensor, storage_p)
   assert(storage_p and storage_p ~= 0, "LongStorage is NULL pointer");
   local cstorage = ffi.cast('THLongStorage*', torch.pointer(tensor:storage()))
   if cstorage ~= nil then
      ffi.C['THLongStorage_free'](cstorage)
   end
   local storage = ffi.cast('THLongStorage*', storage_p)
   tensor:cdata().storage = storage
end

function sendTensor(inputs)
   local size = inputs:size()
   local ttype = inputs:type()
   local i_stg =  tonumber(ffi.cast('intptr_t', torch.pointer(inputs:storage())))
   inputs:cdata().storage = nil
   return {i_stg, size, ttype}
end

function receiveTensor(obj, buffer)
   local pointer = obj[1]
   local size = obj[2]
   local ttype = obj[3]
   if buffer then
      buffer:resize(size)
      assert(buffer:type() == ttype, 'Buffer is wrong type')
   else
      buffer = torch[ttype].new():resize(size)      
   end
   if ttype == 'torch.FloatTensor' then
      setFloatStorage(buffer, pointer)
   elseif ttype == 'torch.LongTensor' then
      setLongStorage(buffer, pointer)
   else
      error('Unknown type')
   end
   return buffer
end

function recordConfusion(confusion, pred, target)
	local pred = pred:float()
	local target = target:float()
	local opt = opt or {}
	if opt.regression then
	    pred:round():clamp(1,5)	    
		pred = pred:view(pred:nElement())
		target = target:view(pred:nElement())
		confusion:batchAdd(pred, target)
	elseif opt.BCE then
	    local predTmp = torch.Tensor(pred:size(2)+1)
	    for j=1,target:size(1) do
			local tind = target[j]:size(1)-target[j]:sum() +1
			local pj = pred[j]
			local lastValue = 0
			for k=1,pj:size(1) do
				predTmp[k] = pj[k] -lastValue
				lastValue = pj[k]
			end

			predTmp[predTmp:size(1)] = 1 - lastValue
			predTmp:clamp(1e-15, 1)
			confusion:add(predTmp, tind)
	   end
	else
		confusion:batchAdd(pred, target)
	end
end

function sanitize(net)
  local list = net:listModules()
  for _,val in ipairs(list) do
		for name,field in pairs(val) do
		   if torch.type(field) == 'cdata' then val[name] = nil end
		   if name == 'homeGradBuffers' then val[name] = nil end
		   if name == 'input_gpu' then val['input_gpu'] = {} end
		   if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
		   if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
		   if (name == 'output' or name == 'gradInput') and (field.new) then
			  val[name] = field.new()
		   end
		end
  end
end

function computeKappa (mat)
	local N = mat:size(1)
	local tmp = torch.range(1, N):view(1, N)
	local tmp1 = torch.range(1, N):view(N, 1)
	local W= tmp:expandAs(mat)-tmp1:expandAs(mat)
	W:cmul(W)
	W:div((N-1)*(N-1))
	local total = mat:sum()
	local row_sum = mat:sum(1)/total
	local col_sum = mat:sum(2)
	local E = torch.cmul(row_sum:expandAs(mat), col_sum:expandAs(mat))
	local kappa = 1 - torch.cmul(W, mat):sum()/torch.cmul(W, E):sum()
	return kappa
end

function saveModel(file, model, param, modelFactory)
	if modelFactory then
	   local modelToSave = modelFactory()
	   local p = modelToSave:getParameters()
	   local param = param or model:getParameters()
	   p:copy(param)
	   torch.save(file, modelToSave)
	else
	   torch.save(file, model)
	end
end

function key2int(keys)
    local result = {}
    local ind = 1
    local int2label = {}
    local label2int = {}

    for i,key in ipairs(keys) do
        if (label2int[key]) then
            result[i] = label2int[key]
        else
            label2int[key] = ind
            result[i] = ind
            table.insert(int2label, key)
            ind = ind + 1
        end
    end
    return result,label2int,int2label
end



function randomSampling(n, pct)
	pct = pct or 0.9
	local nTraining = math.floor(n * pct)
	local nTesting = n - nTraining

	local randIndices = torch.randperm(n)
	local trIndices = randIndices[{{1,nTraining}}]
	local tsIndices
	if nTesting > 0 then
		tsIndices = randIndices[{{nTraining+1, n}}]
	end

	return trIndices, tsIndices
end

