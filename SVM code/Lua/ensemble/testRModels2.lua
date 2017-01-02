require 'torch'
require 'image'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)
CLASSES = {'0', '1', '2', '3', '4'}

local cmd = torch.CmdLine()
opt = cmd:parse(arg)

local loadRFile = function(fName)
  local file = io.open(fName, 'r')
  local skipHead = true
  local nSamples = 0
  for line in file:lines() do
    if not skipHead then
      nSamples = nSamples + 1
    end
    skipHead = false
  end
  local file = io.open(fName, 'r')
  local skipHead = skipHeader or true
  local line
  local j = 0
  local actuals = torch.Tensor(nSamples)
  local preds = torch.Tensor(nSamples)
  for line in file:lines() do
    if not skipHead then
      j = j + 1
      local rowStr = string.split(line, '\t')
      preds[j] = rowStr[1]
      actuals[j] = rowStr[2] + 1
    end
    skipHead = false
  end
  return actuals, preds
end

function computeKappa(mat)
  local N = mat:size(1)
  local tmp = torch.range(1, N):view(1, N)
  local tmp1 = torch.range(1, N):view(N, 1)
  local W= tmp:expandAs(mat) - tmp1:expandAs(mat)
  W:cmul(W)
  W:div((N - 1) * (N - 1))
  local total = mat:sum()
  local row_sum = mat:sum(1) / total
  local col_sum = mat:sum(2)
  local E = torch.cmul(row_sum:expandAs(mat), col_sum:expandAs(mat))
  local kappa = 1 - torch.cmul(W, mat):sum() / torch.cmul(W, E):sum()
  return kappa
end

myf = function(x, preds, labels)
  local confusion = optim.ConfusionMatrix(CLASSES)
  local preds = preds:clone()
  for i = 1, preds:size(1) do
    if preds[i] < x[1] then
      preds[i] = 1
    elseif preds[i] < x[2] then
      preds[i] = 2
    elseif preds[i] < x[3] then
      preds[i] = 3
    elseif preds[i] < x[4] then
      preds[i] = 4
    else
      preds[i] = 5
    end
  end
  confusion:batchAdd(preds, labels)
  confusion:updateValids()
  print(confusion)
  local kappa = computeKappa(confusion.mat)
  return kappa
end



local actuals, preds = loadRFile('kappascan.tsv')
-- local best_x = {0.7842698, 1.3494896, 1.8717345, 2.566831}
local best_x = {0.7, 1.3487674, 1.8685806, 2.5627728}
local starting_kappa = myf(best_x, preds, actuals)
print(starting_kappa)


-- **** BEST *******
-- local best_x = {0.7, 1.3494896, 1.8717345, 2.566831}
-- [[    2433     120      21       0       0]   94.522%   [class: 0]
--  [     117      90      50       0       0]   35.019%   [class: 1]
--  [      65      88     240     135       1]   45.369%   [class: 2]
--  [       3       1       9      68      17]   69.388%   [class: 3]
--  [       0       1       5      21      24]]  47.059%   [class: 4]
--  + average row correct: 58.271359801292% 
--  + average rowUcol correct (VOC measure): 41.619656682014% 
--  + global correct: 81.362211456255%
-- 0.85175094574424
--     0     1     2     3     4 
-- 40397  4407  4459  3397   916 
--          0          1          2          3          4 
-- 0.75401299 0.08225698 0.08322756 0.06340526 0.01709721 
-- LEADERBOARD: 0.85107


-- best score is 0.8540592
-- best cut off
-- 2642 0.7823481
-- 2919 1.3487674
-- 3243 1.8685806
-- 3470 2.5627728
