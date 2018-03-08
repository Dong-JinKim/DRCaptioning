local dbg = require("debugger")
require 'torch'
require 'nn'
require 'nngraph'

require 'densecap.LanguageModel_tLSTM2'--LanguageModel / LanguageModel2  / LanguageModel_union     /LanguageModel_att
require 'densecap.LocalizationLayer_union3'--LocalizationLayer          / LocalizationLayer_union1 / LocalizationLayer_union2
require 'densecap.modules.BoxRegressionCriterion'
require 'densecap.modules.BilinearRoiPooling'
require 'densecap.modules.ApplyBoxTransform'
require 'densecap.modules.LogisticCriterion'
require 'densecap.modules.PosSlicer'
require 'densecap.modules.Pairs2'-- Pairs / Pairs2
require 'densecap.modules.UnionSlicer'

local box_utils = require 'densecap.box_utils'
local utils = require 'densecap.utils'


local DenseCapModel, parent = torch.class('DenseCapModel', 'nn.Module')


function DenseCapModel:__init(opt)
  local net_utils = require 'densecap.net_utils'
  opt = opt or {}  
  opt.cnn_name = utils.getopt(opt, 'cnn_name', 'vgg-16')
  opt.backend = utils.getopt(opt, 'backend', 'cudnn')
  opt.path_offset = utils.getopt(opt, 'path_offset', '')
  opt.dtype = utils.getopt(opt, 'dtype', 'torch.CudaTensor')
  opt.vocab_size = utils.getopt(opt, 'vocab_size')
  opt.std = utils.getopt(opt, 'std', 0.01) -- Used to initialize new layers

  -- For test-time handling of final boxes
  opt.final_nms_thresh = utils.getopt(opt, 'final_nms_thresh', 0.3)

  -- Ensure that all options for loss were specified
  utils.ensureopt(opt, 'mid_box_reg_weight')
  utils.ensureopt(opt, 'mid_objectness_weight')
  utils.ensureopt(opt, 'end_box_reg_weight')
  utils.ensureopt(opt, 'end_objectness_weight')
  utils.ensureopt(opt, 'captioning_weight')
  
  -- Options for RNN
  opt.seq_length = utils.getopt(opt, 'seq_length')
  opt.rnn_encoding_size = utils.getopt(opt, 'rnn_encoding_size', 512)
  opt.rnn_size = utils.getopt(opt, 'rnn_size', 512)
  self.opt = opt -- TODO: this is... naughty. Do we want to create a copy instead?
  
  -- This will hold various components of the model
  self.nets = {}
  
  -- This will hold the whole model
  self.net = nn.Sequential()
  
  -- Load the CNN from disk
  local cnn = net_utils.load_cnn(opt.cnn_name, opt.backend, opt.path_offset)
  
  -- We need to chop the CNN into three parts: conv that is not finetuned,
  -- conv that will be finetuned, and fully-connected layers. We'll just
  -- hardcode the indices of these layers per architecture.
  local conv_start1, conv_end1, conv_start2, conv_end2
  local recog_start, recog_end
  local fc_dim
  if opt.cnn_name == 'vgg-16' then
    conv_start1, conv_end1 = 1, 10 -- these will not be finetuned for efficiency
    conv_start2, conv_end2 = 11, 30 -- these will be finetuned possibly
    recog_start, recog_end = 32, 38 -- FC layers
    opt.input_dim = 512
    opt.output_height, opt.output_width = 7, 7
    fc_dim = 4096
  else
    error(string.format('Unrecognized CNN "%s"', opt.cnn_name))
  end
  
  -- Now that we have the indices, actually chop up the CNN.
  self.nets.conv_net1 = net_utils.subsequence(cnn, conv_start1, conv_end1)
  self.nets.conv_net2 = net_utils.subsequence(cnn, conv_start2, conv_end2)
  self.net:add(self.nets.conv_net1)
  self.net:add(self.nets.conv_net2)
  
  -- Figure out the receptive fields of the CNN
  -- TODO: Should we just hardcode this too per CNN?
  local conv_full = net_utils.subsequence(cnn, conv_start1, conv_end2)
  local x0, y0, sx, sy = net_utils.compute_field_centers(conv_full)
  self.opt.field_centers = {x0, y0, sx, sy}

  self.nets.localization_layer = nn.LocalizationLayer(opt)
  self.net:add(self.nets.localization_layer)
  
  -- Recognition base network; FC layers from VGG.
  -- Produces roi_codes of dimension fc_dim.
  -- TODO: Initialize this from scratch for ResNet?
  self.nets.recog_base = net_utils.subsequence(cnn, recog_start, recog_end)
  self.nets.recog_base2 = net_utils.subsequence(cnn, recog_start, recog_end)
  
  -- Objectness branch; outputs positive / negative probabilities for final boxes
  self.nets.objectness_branch = nn.Linear(fc_dim, 1)
  self.nets.objectness_branch.weight:normal(0, opt.std)
  self.nets.objectness_branch.bias:zero()
  
  -- pairity branch; outputs positive / negative probabilities for final pair-------!!!!!!!!!!!!!!!!
  
  self.nets.pair_mode = nn.View(-1,fc_dim*2)
  self.nets.pair_branch = nn.Linear(fc_dim*2, 1)
  self.nets.pair_branch.weight:normal(0, opt.std)----!!!
  self.nets.pair_branch.bias:zero()
  
  
  -- Final box regression branch; regresses from RPN boxes to final boxes
  self.nets.box_reg_branch = nn.Linear(fc_dim, 4)
  self.nets.box_reg_branch.weight:zero()
  self.nets.box_reg_branch.bias:zero()

  -- Set up LanguageModel
  local lm_opt = {
    vocab_size = opt.vocab_size,
    input_encoding_size = opt.rnn_encoding_size,
    rnn_size = opt.rnn_size,
    seq_length = opt.seq_length,
    idx_to_token = opt.idx_to_token,
    image_vector_dim=fc_dim,
  }
  self.nets.language_model = nn.LanguageModel(lm_opt)

  self.nets.recog_net = self:_buildRecognitionNet()
  self.net:add(self.nets.recog_net)

  -- Set up Criterions
  self.crits = {}
  self.crits.objectness_crit = nn.LogisticCriterion()
  self.crits.pairity_crit = nn.LogisticCriterion()
  self.crits.box_reg_crit = nn.BoxRegressionCriterion(opt.end_box_reg_weight)
  self.crits.lm_crit = nn.TemporalCrossEntropyCriterion()  
  self.crits.cls_crit = nn.TemporalCrossEntropyCriterion()

  self:training()
  self.finetune_cnn = false
end


function DenseCapModel:_buildRecognitionNet()
  local roi_feats = nn.Identity()()
  local roi_boxes = nn.Identity()()
  local gt_boxes = nn.Identity()()
  local gt_labels = nn.Identity()()
  local union = nn.Identity()()
  local spatial = nn.Identity()()
  local idx = nn.Identity()()

  
  local roi_codes = self.nets.recog_base(roi_feats)
  
  
  
  local objectness_scores = self.nets.objectness_branch(roi_codes)---!!!333
  

  local pos_roi_codes = nn.PosSlicer(){roi_codes, gt_boxes}---gt_labels 
  local pos_roi_boxes = nn.PosSlicer(){roi_boxes, gt_boxes}
  
  --local subjobj = nn.UnionSlicer(){pos_roi_codes,idx}--FC7 feat recogbase
  local subjobj = nn.UnionSlicer(){roi_feats,idx}--CONV5 feat all3
  
  
  
  local final_box_trans = self.nets.box_reg_branch(pos_roi_codes)----------------!!! it should be not only pos box but also pos pair
  
  local final_boxes = nn.ApplyBoxTransform(){pos_roi_boxes, final_box_trans}
  
  
  --local Pair_output = nn.Pairs(){pos_roi_codes,union,final_boxes}
  
  --local lm_input = {pos_roi_codes, gt_labels}
  local lm_input = {union, gt_labels ,spatial,subjobj}----!!!!!!!!!!!!!!!!!! 2222233333?????
  --local lm_input = {union, gt_labels ,spatial}
  
  --local lm_input = {Pair_output, gt_labels}
  
  local lm_output = self.nets.language_model(lm_input)

  -- Annotate nodes
  roi_codes:annotate{name='recog_base'}
  objectness_scores:annotate{name='objectness_branch'}
  pos_roi_codes:annotate{name='code_slicer'}
  pos_roi_boxes:annotate{name='box_slicer'}
  final_box_trans:annotate{name='box_reg_branch'}
  local inputs = {roi_feats, roi_boxes, gt_boxes, gt_labels , union ,spatial, idx}----!!!333334444
  local outputs = {
    objectness_scores,
    pos_roi_boxes, final_box_trans, final_boxes,
    lm_output,
    gt_boxes, gt_labels--,pos_roi_codes,--union,--pairity_scores,
  }
  local mod = nn.gModule(inputs, outputs)
  mod.name = 'recognition_network'
  return mod
end


function DenseCapModel:training()
  parent.training(self)
  self.net:training()
end


function DenseCapModel:evaluate()
  parent.evaluate(self)
  self.net:evaluate()
end


--[[
Set test-time parameters for this DenseCapModel.

Input: Table with the following keys:
- rpn_nms_thresh: NMS threshold for region proposals in the RPN; default is 0.7.
- final_nms_thresh: NMS threshold for final predictions; default is 0.3.
- num_proposals: Number of proposals to use; default is 1000
--]]
function DenseCapModel:setTestArgs(kwargs)
  self.nets.localization_layer:setTestArgs{
    nms_thresh = utils.getopt(kwargs, 'rpn_nms_thresh', 0.7),
    max_proposals = utils.getopt(kwargs, 'num_proposals', 50)---!!!! originally 1000
  }
  self.opt.final_nms_thresh = utils.getopt(kwargs, 'final_nms_thresh', 0.3)
end


--[[
Convert this DenseCapModel to a particular datatype, and convert convolutions
between cudnn and nn.
--]]
function DenseCapModel:convert(dtype, use_cudnn)
  self:type(dtype)
  if cudnn and use_cudnn ~= nil then
    local backend = nn
    if use_cudnn then
      backend = cudnn
    end
    cudnn.convert(self.net, backend)
    cudnn.convert(self.nets.localization_layer.nets.rpn, backend)
  end
end


--[[
Run the model forward.

Input:
- image: Pixel data for a single image of shape (1, 3, H, W)

After running the model forward, we will process N regions from the
input image. At training time we have access to the ground-truth regions
for that image, and assume that there are P ground-truth regions. We assume
that the language model has a vocabulary of V elements (including the END
token) and that all captions have been padded to a length of L.

Output: A table of
- objectness_scores: Array of shape (N, 1) giving (final) objectness scores
  for boxes.
- pos_roi_boxes: Array of shape (P, 4) at training time and (N, 4) at test-time
  giving the positions of RoI boxes in (xc, yc, w, h) format.
- final_box_trans: Array of shape (P, 4) at training time and (N, 4) at
  test-time giving the transformations applied on top of the region proposal
  boxes by the final box regression.
- final_boxes: Array of shape (P, 4) at training time and (N, 4) at test-time
  giving the coordinates of the final output boxes, in (xc, yc, w, h) format.
- lm_output: At training time, an array of shape (P, L+2, V) giving log
  probabilities (the +2 is two extra time steps for CNN input and END token).
  At test time, an array of shape (N, L) where each element is an integer in
  the range [1, V] giving sampled captions.
- gt_boxes: At training time, an array of shape (P, 4) giving ground-truth
  boxes corresponding to the sampled positives. At test time, an empty tensor.
- gt_labels: At training time, an array of shape (P, L) giving ground-truth
  sequences for sampled positives. At test-time, and empty tensor.
--]]
function DenseCapModel:updateOutput(input)
  -- Make sure the input is (1, 3, H, W)
  
  assert(input:dim() == 4 and input:size(1) == 1 and input:size(2) == 3)
  local H, W = input:size(3), input:size(4)
  self.nets.localization_layer:setImageSize(H, W)

  if self.train then
    assert(not self._called_forward,
      'Must call setGroundTruth before training-time forward pass')
    self._called_forward = true
  end
  
  self.output = self.net:forward(input)
  
  -- At test-time, apply NMS to final boxes
  local verbose = false
  if verbose then
    print(string.format('before final NMS there are %d boxes', self.output[4]:size(1)))
    print(string.format('Using NMS threshold of %f', self.opt.final_nms_thresh))
  end
  if not self.train and self.opt.final_nms_thresh > 0 then
    -- We need to apply the same NMS mask to the final boxes, their
    -- objectness scores, and the output from the language model
    local final_boxes_float = self.output[4]:float()
    local class_scores_float = self.output[1]:float()
    --local pairity_scores_float = self.output[9]:float()------!!!!!! $$$
    
    local TRANSF = false----!!!%%%
    local transfer
    local POS
    local lm_output_float
    if TRANSF then
      POS = self.output[5][3]
      transfer = self.output[5][2]
      lm_output_float = self.output[5][1]:float()
    else
      lm_output_float = self.output[5]:float()
    end
    
    local boxes_scores = torch.FloatTensor(final_boxes_float:size(1), 5)
    local boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(final_boxes_float)
    boxes_scores[{{}, {1, 4}}]:copy(boxes_x1y1x2y2)
    boxes_scores[{{}, 5}]:copy(class_scores_float[{{}, 1}])
    
    local idx = box_utils.nms(boxes_scores, self.opt.final_nms_thresh)-----!!!!!
    --dbg()
    self.output[4] = final_boxes_float:index(1, idx):typeAs(self.output[4])
    self.output[1] = class_scores_float:index(1, idx):typeAs(self.output[1])
    
    --self.output[9] = pairity_scores_float:index(1, idx):typeAs(self.output[9])-------!!!!! $$$
    local idx_
    local SHUFFLE=true
    if SHUFFLE then
      --idx_ = idx:size(1)*(idx:size(1)-1):zero()
      idx_ = torch.LongTensor(0)
      for i = 1,idx:size(1) do----------------------------------------------------!!!!!!!!!!!!!!!!!!!!
        for j = 1,idx:size(1) do
          if not(i==j) then
            if i>j then
              idx_ = idx_:cat(torch.LongTensor({ (idx[i]-1)*(boxes_scores:size(1)-1) +  idx[j]   }),1)----!!!!qualified?
            else--i<j
              idx_ = idx_:cat(torch.LongTensor({ (idx[i]-1)*(boxes_scores:size(1)-1) +  idx[j]-1 }),1)----!!!!qualified?
            end
          
          end
        end
      end
      
      --dbg()
      if ##idx_ == 0 then---no box detected
        --  dbg()
        if TRANSF then
          self.output[5] = {idx_:new(idx_:size()):typeAs(self.output[5]),transfer,POS}
        else
          self.output[5] = idx_:new(idx_:size()):typeAs(self.output[5])
        end
      else
        if TRANSF then
          self.output[5] = {lm_output_float:index(1, idx_ ):typeAs(self.output[5][1]), transfer:index(1,idx_), POS:index(1,idx_)}
        else
          self.output[5] = lm_output_float:index(1, idx_ ):typeAs(self.output[5])
        end
      end
    
    else
    
      self.output[5] = lm_output_float:index(1,idx):typeAs(self.output[5])
    end
      -- TODO: In the old StnDetectionModel we also applied NMS to the
      -- variables dumped by the LocalizationLayer. Do we want to do that?
  end
  return self.output
end

--[[
Run a test-time forward pass, plucking out only the relevant outputs.

Input: Tensor of shape (1, 3, H, W) giving pixels for an input image.

Returns:
- final_boxes: Tensor of shape (N, 4) giving coordinates of output boxes
  in (xc, yc, w, h) format.
- objectness_scores: Tensor of shape (N, 1) giving objectness scores of
  those boxes.
- captions: Array of length N giving output captions, decoded as strings.
--]]
function DenseCapModel:forward_test(input)
  self:evaluate()
  local output = self:forward(input)
  local final_boxes = output[4]
  local objectness_scores = output[1]
  
  local TRANSF = false---!!!%%%
  local transfer
  local pos
  local captions
  if TRANSF then
    captions = output[5][1]
    transfer = output[5][2]
    POS = output[5][3]
  else
    captions = output[5]
  end
  
  local captions = self.nets.language_model:decodeSequence(captions)
  if TRANSF then
    return final_boxes, objectness_scores, {captions,transfer,POS}
  else
    return final_boxes, objectness_scores, captions
  end
end


function DenseCapModel:setGroundTruth(gt_boxes, gt_labels)
  self.gt_boxes = gt_boxes
  self.gt_labels = gt_labels
  self._called_forward = false
  self.nets.localization_layer:setGroundTruth(gt_boxes, gt_labels)
end


function DenseCapModel:backward(input, gradOutput)
  -- Manually backprop through part of the network
  -- self.net has 4 elements:
  -- (1) CNN part 1        (2) CNN part 2
  -- (3) LocalizationLayer (4) Recognition network
  -- We always backprop through (3) and (4), and only backprop through
  -- (2) when finetuning; we never backprop through (1).
  -- Note that this means we break the module API in this method, and don't
  -- actually return gradients with respect to our input.

  local end_idx = 3
  if self.finetune_cnn then end_idx = 2 end
  local dout = gradOutput
  for i = 4, end_idx, -1 do
    local layer_input = self.net:get(i-1).output
    dout = self.net:get(i):backward(layer_input, dout)
  end

  self.gradInput = dout
  return self.gradInput
end


--[[
We naughtily override the module's getParameters method, and return:
- params: Flattened parameters for the RPN and recognition network
- grad_params: Flattened gradients for the RPN and recognition network
- cnn_params: Flattened portion of the CNN parameters that will be fine-tuned
- grad_cnn_params: Flattened gradients for the portion of the CNN that will
  be fine-tuned.
--]]
function DenseCapModel:getParameters()
  local cnn_params, grad_cnn_params = self.net:get(2):getParameters()
  local fakenet = nn.Sequential()
  fakenet:add(self.net:get(3))
  fakenet:add(self.net:get(4))
  local params, grad_params = fakenet:getParameters()
  return params, grad_params, cnn_params, grad_cnn_params
end


function DenseCapModel:clearState()
  self.net:clearState()
  for k, v in pairs(self.crits) do
    if v.clearState then
      v:clearState()
    end
  end
end

--[[
Perform a (training-time) forward pass to compute output and loss,
and a backward pass to compute gradients.

This is a nonstandard method, but it allows the DenseCapModel to
have control over its own Criterions.

Input: data is table with the following keys:
- image: 1 x 3 x H x W array of pixel data
- gt_boxes: 1 x B x 4 array of ground-truth object boxes
  TODO: What format are the boxes?
- gt_labels: 1 x B x L array of ground-truth sequences for boxes
--]]
function DenseCapModel:forward_backward(data)
  
  self:training()
  -- Run the model forward
  self:setGroundTruth(data.gt_boxes, data.gt_labels)
  local out = self:forward(data.image)
  
  -- Pick out the outputs we care about
  local objectness_scores = out[1]
  local pos_roi_boxes = out[2]
  local final_box_trans = out[3]
  local lm_output = out[5]
  local gt_boxes = out[6]
  local gt_labels = out[7]
  local gt_labels1 = gt_labels---!!!!
  
  --local gt_pairs = out[9]--$$$

  local num_boxes = objectness_scores:size(1)
  local num_pos = pos_roi_boxes:size(1)
  
  -- Compute final objectness loss and gradient
  local objectness_labels = torch.LongTensor(num_boxes):zero()
  objectness_labels[{{1, num_pos}}]:fill(1)
  
  local end_objectness_loss = self.crits.objectness_crit:forward(
                                         objectness_scores, objectness_labels)
                                       
  end_objectness_loss = end_objectness_loss * self.opt.end_objectness_weight
  local grad_objectness_scores = self.crits.objectness_crit:backward(
                                      objectness_scores, objectness_labels)
  grad_objectness_scores:mul(self.opt.end_objectness_weight)


  --local pairity_labels = torch.LongTensor(num_boxes/2):zero()
  --pairity_labels[{{1, num_pos}}]:fill(1)-------------------------------------------!!!!!!!
  
  --local pairity_loss = self.crits.pairity_crit:forward(pairity_scores, gt_pairs)----!!!!
                                       
  --pairity_loss = pairity_loss * self.opt.end_objectness_weight
  --local grad_pairity_scores = self.crits.pairity_crit:backward(
   --                                   pairity_scores, gt_pairs)
  --grad_pairity_scores:mul(self.opt.end_objectness_weight)
  
  
  -- Compute box regression loss; this one multiplies by the weight inside
  -- the criterion so we don't do it manually.
  local end_box_reg_loss = self.crits.box_reg_crit:forward(
                                {pos_roi_boxes, final_box_trans},
                                gt_boxes)
  local din = self.crits.box_reg_crit:backward(
                         {pos_roi_boxes, final_box_trans},
                         gt_boxes)
  local grad_pos_roi_boxes, grad_final_box_trans = unpack(din)

  -- Compute captioning loss
  
  --label_idx = gt_pairs
  --gt_labels = gt_labels:index(1,label_idx) --------------------------------------!!!!!!!!!!!!!!!
  
  local grad_lm_output=torch.CudaLongTensor(0)--initiallize
  local captioning_loss=0
  local part_loss = nil
  
  if lm_output == 0 then--- if no matched points, return 0
    captioning_loss = 0
    grad_lm_output = 0--torch.CudaLongTensor(0)
  else
    
    if #lm_output==2 then-- if cls branch is added !!!!!!!!!!!!!!
      
      part_output = lm_output[2]
      lm_output = lm_output[1]
      

      --local part_target = gt_labels.new(gt_labels:size(1),16):zero()---!!! clsATT
      --part_target[{{},{1,15}}]:copy(gt_labels[{{},{16,30}}])
      
      local part_target = gt_labels.new(gt_labels:size(1),17):zero()---!!! MTL*
      part_target[{{},{2,16}}]:copy(gt_labels[{{},{16,30}}])---!!!! MTL
      --part_target[{{},{1,15}}]:copy(gt_labels[{{},{16,30}}])---!!! MTL2
      --part_target[{{},1}]:fill(1)---!!!! MTL_real
      --part_target[{{},{2,16}}]:copy(gt_labels[{{},{16,30}}])---!!!! MTL_real
      
      
      gt_labels1 = gt_labels[{{},{1,15}}]----!!!!!
      
      part_loss = self.crits.cls_crit:forward(part_output, part_target)
      part_loss = part_loss * 0.1
      grad_cls_output = self.crits.cls_crit:backward(part_output, part_target)
      grad_cls_output:mul(0.1)
    end
    
    local target = self.nets.language_model:getTarget(gt_labels1)--!!!!
    captioning_loss = self.crits.lm_crit:forward(lm_output, target)
    
    
    --local length = (16-target:eq(0):sum(2)):long()
   
    --local weight_LUT = torch.FloatTensor{0.01, 0.01, 0.43, 0.46, 0.51, 0.59, 0.72, 0.84, 0.93, 1.07, 1.29, 1.61, 5.54, 0.01, 0.01}
    
    --dbg()
    --local length_weight = (1+ (5.5-length):exp())/2
    --dbg()
    --local length_weight = weight_LUT:index(1,length:view(-1))
    
    
    captioning_loss = captioning_loss * self.opt.captioning_weight
    grad_lm_output = self.crits.lm_crit:backward(lm_output, target)
    grad_lm_output:mul(self.opt.captioning_weight)
    
  end

  local ll_losses = self.nets.localization_layer.stats.losses
  local losses = {
    mid_objectness_loss=ll_losses.obj_loss_pos + ll_losses.obj_loss_neg,
    mid_box_reg_loss=ll_losses.box_reg_loss,
    end_objectness_loss=end_objectness_loss,
    end_box_reg_loss=end_box_reg_loss,
    captioning_loss=captioning_loss,
    part_loss = part_loss,-----------------------------!!!!!!!!!!!!!!!
  }
  local total_loss = 0
  for k, v in pairs(losses) do
    total_loss = total_loss + v
  end
  losses.total_loss = total_loss

  -- Run the model backward
  local grad_out = {}
  grad_out[1] = grad_objectness_scores
  grad_out[2] = grad_pos_roi_boxes
  grad_out[3] = grad_final_box_trans
  grad_out[4] = out[4].new(#out[4]):zero()
  
  if part_loss then
    grad_out[5] = {grad_lm_output, grad_cls_output}
  else
    grad_out[5] = grad_lm_output
  end
  
  
  grad_out[6] = gt_boxes.new(#gt_boxes):zero()
  grad_out[7] = gt_labels.new(#gt_labels):zero()
  --grad_out[8] = out[8].new(#out[8]):zero()--!!!!!88888
  --grad_out[9] = gt_pairs.new(#gt_pairs):zero()--!!!$$$
  self:backward(input, grad_out)
  
  return losses
end
