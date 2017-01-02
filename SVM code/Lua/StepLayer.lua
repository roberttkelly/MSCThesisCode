local StepLayer, parent = torch.class('nn.StepLayer', 'nn.Module')

function StepLayer:__init(thr, alpha)
   parent.__init(self)
   assert(#thr == #alpha, "# of thr must equal # of alpha")
   self.bias= torch.Tensor(thr):view(1, #thr)
   self.weight= torch.Tensor(alpha):view(1, #alpha)
   self.gradBias= torch.Tensor(thr):view(1, #thr)
   self.gradWeight= torch.Tensor(alpha):view(1, #alpha)
   self.tmp1 = torch.Tensor()
   self.tmp2 = torch.Tensor()
   self.tmp3 = torch.Tensor()
   self.tmp4 = torch.Tensor()
   self.one2 = torch.ones(#alpha, 1)
end

function StepLayer:updateOutput(input)
   local nBatch = input:size(1)
   local nDim = input:dim()
   assert (input:size(2) == 1, "only take one dimension as input")
   local tmp1 = self.tmp1
   local tmp2 = self.tmp2
   tmp1:resize(nBatch, self.weight:size(2))
   tmp2:resize(nBatch, self.weight:size(2)):fill(1)
   self.output:resize(nBatch, 1)
   torch.add(tmp1, input:expandAs(tmp1), -1, self.bias:expandAs(tmp1) )
   tmp1:cmul(self.weight:expandAs(tmp1)):mul(-1)
   tmp1:exp():add(1)
   tmp2:cdiv(tmp1)
   self.output:mm(tmp2, self.one2)
   self.output:add(1)
   return self.output
end

function StepLayer:updateGradInput(input, gradOutput)
   local nBatch = input:size(1)
   local nDim = input:dim()
   assert (input:size(2) == 1, "only take one dimension as input")
   local tmp2 = self.tmp2
   local tmp3 = self.tmp3
   tmp3:resizeAs(tmp2):copy(tmp2)
   tmp3:mul(-1):add(1):cmul(self.weight:expandAs(tmp3))
   tmp3:cmul(tmp2)
   self.gradInput:resizeAs(input)
   self.gradInput:mm(tmp3, self.one2)
   self.gradInput:cmul(gradOutput)
   return self.gradInput
end

function StepLayer:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local tmp2 = self.tmp2
   local tmp3 = self.tmp3
   local tmp4 = self.tmp4
   tmp4:resizeAs(tmp2)
   tmp4:add(input:expandAs(tmp2), -1, self.bias:expandAs(tmp4))
   tmp3:resizeAs(tmp2):copy(tmp2)
   tmp3:mul(-1):add(1)
   tmp3:cmul(tmp2)
   tmp4:cmul(tmp3)
   self.gradWeight:addmm(scale, gradOutput:t(), tmp4)
   tmp3:cmul(self.weight:expandAs(tmp3)):mul(-1)
   self.gradBias:addmm(scale, gradOutput:t(), tmp3)
end
