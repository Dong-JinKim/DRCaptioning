local dbg = require("debugger")
local layer, parent = torch.class('nn.Pairs', 'nn.Module')
local box_utils = require 'densecap.box_utils'
require 'densecap.modules.BoxIoU'


function layer:__init()
  parent.__init(self)
  self.grad_features = torch.Tensor()
  self.grad_gt_features = torch.Tensor()
end

function layer:updateOutput(input)--------- we need to select only relevant pairs!
  
  local feat = input[1]
  local box_idxs = input[2]
  local final_boxes = input[3]
  local B, D = feat:size(1),feat:size(2)
  local M=32--mask size 
  
  local feat_pairs = torch.CudaTensor(0)
  local label_idx = torch.LongTensor(0)
  local Spatial = torch.CudaTensor(0)
  local tmp = torch.CudaTensor(1,6)
  local continue = false
  
  local IOU = nn.BoxIoU():forward{final_boxes:view(1,-1,4):float() ,final_boxes:view(1,-1,4):float()}
  for i =1,B do---subj
    for j = 1,B do----obj
      
      ----conditions to pass
      if ##box_idxs == 0  then --for test
        if i==j then
          continue = true---- if i==j then pass
        else
          continue = false
        end
      elseif not( (torch.ceil(box_idxs[i]/2)==torch.ceil(box_idxs[j]/2) and box_idxs[i]%2==1 and box_idxs[j]%2==0) ) then--for training data
        continue = true
      else
        continue = false
      end   
      
      if not continue then
        local xs = final_boxes[{i,1}]
        local ys = final_boxes[{i,2}]
        local ws = final_boxes[{i,3}]
        local hs = final_boxes[{i,4}]
        local xo = final_boxes[{j,1}]
        local yo = final_boxes[{j,2}]
        local wo = final_boxes[{j,3}]
        local ho = final_boxes[{j,4}]
        --dbg()

        
        tmp[{1,1}] = (xs-xo)/torch.sqrt(ws*hs)
        tmp[{1,2}] = (ys-yo)/torch.sqrt(ws*hs)
        tmp[{1,3}] = torch.sqrt(wo*ho)/torch.sqrt(ws*hs)
        tmp[{1,4}] = ws/hs
        tmp[{1,5}] = wo/ho
        tmp[{1,6}] = IOU[{1,i,j}]
        --tmp[{1,7}] = xs/ws
        --tmp[{1,8}] = ys/hs
        --tmp[{1,9}] = xo/wo
        --tmp[{1,10}] = yo/ho

        --dbg()
        Spatial = Spatial:cat(tmp , 1)--!!!!!!!!!!!!
        feat_pairs = feat_pairs:cat( (torch.cat(feat[{i,{}}], feat[{j,{}}],1 ):view(1,-1))   ,1)-- concat feat pairs
        label_idx = label_idx:cat(torch.LongTensor({ i }),1)
      end
    end
  end
  --
  --self.output[{gt_pairs:nonzero()}]
  --dbg()
  --if ##box_idxs == 0 then
    --dbg()
  --end
  self.output = { feat_pairs , label_idx,Spatial}
  
  return self.output
end

function layer:updateGradInput(input, gradOutput)
  local feat = input[1]
  local box_idxs = input[2]
  
  local feat_grad = feat.new(#feat):zero()
  local box_grad = box_idxs.new(#box_idxs):zero()
  
  self.gradInput = {feat_grad,box_grad}-------------no back prop!!!!!!!!!!
  return self.gradInput
end


