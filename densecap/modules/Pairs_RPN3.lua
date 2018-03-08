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
  
  local label_idx = torch.LongTensor(0)
  local union_boxes = boxes.new(0)
  
  local Spatial = torch.CudaTensor(0)
  local spatial = torch.CudaTensor(1,8)
  
  local continue = false
  local IOU = nn.BoxIoU():forward{boxes:view(1,-1,4) ,boxes:view(1,-1,4)}
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
        
        
        local union = box_utils.x1y1x2y2_to_xcycwh( torch.cat(x1y1,x2y2) )--compute union of box
        
        ----------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        local X1_subj = torch.ceil((subj_x1y1x2y2[1][1]-x1y1[1][1]) * 7 / (x2y2[1][1]- x1y1[1][1]))--location in the featuremap
        local X2_subj = torch.ceil((subj_x1y1x2y2[1][3]-x1y1[1][1]) * 7 / (x2y2[1][1]- x1y1[1][1]))
        
        local Y1_subj = torch.ceil((subj_x1y1x2y2[1][2]-x1y1[1][2]) * 7 / (x2y2[1][2]- x1y1[1][2]))
        local Y2_subj = torch.ceil((subj_x1y1x2y2[1][4]-x1y1[1][2]) * 7 / (x2y2[1][2]- x1y1[1][2]))
        
        local X1_obj = torch.ceil((obj_x1y1x2y2[1][1]-x1y1[1][1]) * 7 / (x2y2[1][1]- x1y1[1][1]))--location in the featuremap
        local X2_obj = torch.ceil((obj_x1y1x2y2[1][3]-x1y1[1][1]) * 7 / (x2y2[1][1]- x1y1[1][1]))
        
        local Y1_obj = torch.ceil((obj_x1y1x2y2[1][2]-x1y1[1][2]) * 7 / (x2y2[1][2]- x1y1[1][2]))
        local Y2_obj = torch.ceil((obj_x1y1x2y2[1][4]-x1y1[1][2]) * 7 / (x2y2[1][2]- x1y1[1][2]))
        
        --if 0 -> +1(to 1) !!!
        if X1_subj==0 then X1_subj=X1_subj+1 end
        if Y1_subj==0 then Y1_subj=Y1_subj+1 end
        if X1_obj==0  then X1_obj=X1_obj+1  end
        if Y1_obj==0  then Y1_obj=Y1_obj+1  end
        spatial[{{}}] = torch.CudaTensor{X1_subj,X2_subj,Y1_subj,Y2_subj,X1_obj,X2_obj,Y1_obj,Y2_obj}
        
        Spatial = Spatial:cat(spatial , 1)--!!!!!!!!!!!!
        union_boxes = union_boxes:cat(union,1)
        label_idx = label_idx:cat(torch.LongTensor({ i,j }),2)
      end
    end
  end

  self.output = { union_boxes , label_idx,Spatial}
  return union_boxes,label_idx,Spatial--self.output
end

function layer:updateGradInput(input, gradOutput)
  local boxes = input[1]
  local box_idxs = input[2]
  
  local feat_grad = boxes.new(#boxes):zero()
  local box_grad = box_idxs.new(#box_idxs):zero()
  
  self.gradInput = {feat_grad,box_grad}-------------no back prop!!!!!!!!!!
  return self.gradInput
end


