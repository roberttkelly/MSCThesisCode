local imgJitter = require './ImgJitter'

local JitterLayer, parent = torch.class('nn.JitterLayer', 'nn.Module')

function JitterLayer:__init(outWidth, outHeight, jitterFunc)
   parent.__init(self)
   self.outWidth= outWidth
   self.outHeight= outHeight
   self.jitterFunc = jitterFunc or imgJitter.getJDTransform
   self.gradInput = nil
end

function JitterLayer:updateOutput(input)
   local nBatch = input:size(1)
   local nDim = input:dim()
   if (nDim == 3) then
   	  nBatch = 1
   end
   local coeff = torch.Tensor(nBatch,2,3)
   local width = input:size(nDim)
   local height = input:size(nDim-1)
   for i = 1, nBatch do
   		coeff[i] = self.jitterFunc(width, height, self.outWidth, self.outHeight)
   end
   self.output = {input, coeff}
   return self.output
end

