local dbg = require("debugger")
local layer, parent = torch.class('nn.Pairs_RPN', 'nn.Module')
local box_utils = require 'densecap.box_utils'
require 'densecap.modules.BoxIoU'


function layer:__init()
  parent.__init(self)
  self.grad_features = torch.Tensor()
  self.grad_gt_features = torch.Tensor()
end

function layer:updateOutput(input)--------- we need to select only relevant pairs!
  
  local box_idxs = input[2]
  local boxes = input[1]
  local B = boxes:size(1)
  local M = 32
  
  local label_idx = torch.LongTensor(0)
  local union_boxes = boxes.new(0)
  
  local Masks = torch.CudaTensor(0)
  
  local mask_tmp = torch.CudaTensor(1,2,M,M):zero()-- prepare 32*32 mask
  
  local continue = false
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
        
        local subj_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(boxes[{{i}}])--box coordinates
        local obj_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(boxes[{{j}}])
        
        local x1y1 = torch.cat(subj_x1y1x2y2,obj_x1y1x2y2,1 ):min(1)[{{},{1,2}}]
        local x2y2 = torch.cat(subj_x1y1x2y2,obj_x1y1x2y2,1 ):max(1)[{{},{3,4}}]



        
        
        local X = torch.min(torch.Tensor({subj_x1y1x2y2[1][1],obj_x1y1x2y2[1][1]}))-1----origin
        local Y = torch.min(torch.Tensor({subj_x1y1x2y2[1][2],obj_x1y1x2y2[1][2]}))-1
        
        subj_x1y1x2y2:csub(torch.CudaTensor({X,Y,X,Y}))--normalized(?) coordinate
        obj_x1y1x2y2:csub(torch.CudaTensor({X,Y,X,Y}))
        
        
        local W = torch.max(torch.Tensor({subj_x1y1x2y2[1][3],obj_x1y1x2y2[1][3]}))-- max size of box
        local H = torch.max(torch.Tensor({subj_x1y1x2y2[1][4],obj_x1y1x2y2[1][4]}))
        
        assert(W>0 and H>0)
        
        mask_tmp:zero()
        
        mask_tmp[{1,1, {torch.ceil(subj_x1y1x2y2[1][1]*M/W), torch.ceil(subj_x1y1x2y2[1][3]*M/W)}, 
          {torch.ceil(subj_x1y1x2y2[1][2]*M/H), torch.ceil(subj_x1y1x2y2[1][4]*M/H)} }] = 1--subject mask
        mask_tmp[{1,2, {torch.ceil(obj_x1y1x2y2[1][1]*M/W), torch.ceil(obj_x1y1x2y2[1][3]*M/W)}, 
          {torch.ceil(obj_x1y1x2y2[1][2]*M/H), torch.ceil(obj_x1y1x2y2[1][4]*M/H)}}] = 1--object mask
        
        Masks = Masks:cat(mask_tmp , 1)--!!!!!!!!!!!!
        
        
        
        
        
        local union = box_utils.x1y1x2y2_to_xcycwh( torch.cat(x1y1,x2y2) )--compute union of box
        
        union_boxes = union_boxes:cat(union,1)
        label_idx = label_idx:cat(torch.LongTensor({ i,j }),2)
      end
    end
  end
  self.output = { union_boxes , label_idx,Masks}
  return union_boxes , label_idx,Masks--self.output
end

function layer:updateGradInput(input, gradOutput)
  local boxes = input[1]
  local box_idxs = input[2]
  
  local feat_grad = boxes.new(#boxes):zero()
  local box_grad = box_idxs.new(#box_idxs):zero()
  
  self.gradInput = {feat_grad,box_grad}-------------no back prop!!!!!!!!!!
  return self.gradInput
end


