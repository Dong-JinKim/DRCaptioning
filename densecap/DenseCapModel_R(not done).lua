local dbg = require("debugger")
require 'torch'
require 'nn'
require 'nngraph'

require 'densecap.LanguageModel_R'
require 'densecap.DiscriminatorModel'
require 'densecap.LocalizationLayer'
require 'densecap.modules.BoxRegressionCriterion'
require 'densecap.modules.BilinearRoiPooling'
require 'densecap.modules.ApplyBoxTransform'
require 'densecap.modules.LogisticCriterion'
require 'densecap.modules.PosSlicer'
require 'densecap.modules.Pairs'

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


  --Set up Discriminator for GAN
  self.nets.discriminator_model = nn.DiscriminatorModel(lm_opt)--------------dircriminator
  
  self.nets.recog_net = self:_buildRecognitionNet()
  self.net:add(self.nets.recog_net)

  -- Set up Criterions
  self.crits = {}
  self.crits.objectness_crit = nn.LogisticCriterion()
  self.crits.pairity_crit = nn.LogisticCriterion()
  self.crits.box_reg_crit = nn.BoxRegressionCriterion(opt.end_box_reg_weight)
  self.crits.lm_crit = nn.BCECriterion()----GAN!!!!
  self.crits.matching_crit = nn.MSECriterion()--!!!!

  self:training()
  self.finetune_cnn = false
end


function DenseCapModel:_buildRecognitionNet()
  local roi_feats = nn.Identity()()
  local roi_boxes = nn.Identity()()
  local gt_boxes = nn.Identity()()
  local gt_labels = nn.Identity()()
  local gt_pairs = nn.Identity()()------!!!!

  local roi_codes = self.nets.recog_base(roi_feats)
  local objectness_scores = self.nets.objectness_branch(roi_codes)
  
  

  local pos_roi_codes = nn.PosSlicer(){roi_codes, gt_labels}
  local pos_roi_boxes = nn.PosSlicer(){roi_boxes, gt_boxes}
  
  
  
  local final_box_trans = self.nets.box_reg_branch(pos_roi_codes)----------------!!! it should be not only pos box but also pos pair
  
  local final_boxes = nn.ApplyBoxTransform(){pos_roi_boxes, final_box_trans}
  
  local Pair_output = nn.Pairs(){pos_roi_codes,gt_pairs}----------------------------!!!!
  --local pair_codes = Pair_output[1]
  --local label_idx = Pair_output[2]
  
  --local pairity_scores = self.nets.pair_branch(pair_codes)----!!!
  
  local lm_input = {Pair_output, gt_labels}----!!!!!!!!!!!!!!!!!!
  local lm_output = self.nets.language_model(lm_input)

  -- Annotate nodes
  roi_codes:annotate{name='recog_base'}
  objectness_scores:annotate{name='objectness_branch'}
  --pairity_scores:annotate{name='pair_branch'}--------------!!!!!
  pos_roi_codes:annotate{name='code_slicer'}
  pos_roi_boxes:annotate{name='box_slicer'}
  final_box_trans:annotate{name='box_reg_branch'}

  local inputs = {roi_feats, roi_boxes, gt_boxes, gt_labels, gt_pairs}----!!!
  local outputs = {
    objectness_scores,
    pos_roi_boxes, final_box_trans, final_boxes,
    lm_output,
    gt_boxes, gt_labels,Pair_output,gt_pairs,--pairity_scores,
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
    max_proposals = utils.getopt(kwargs, 'num_proposals', 1000)
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
    cudnn.convert(self.nets.discriminator_model,backend)
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
    local pairity_scores_float = self.output[9]:float()------!!!!!!
    local lm_output_float = self.output[5]:float()
    local boxes_scores = torch.FloatTensor(final_boxes_float:size(1), 5)
    local boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(final_boxes_float)
    boxes_scores[{{}, {1, 4}}]:copy(boxes_x1y1x2y2)
    boxes_scores[{{}, 5}]:copy(class_scores_float[{{}, 1}])
    
    local idx = box_utils.nms(boxes_scores, self.opt.final_nms_thresh)-----!!!!!
    --dbg()
    self.output[4] = final_boxes_float:index(1, idx):typeAs(self.output[4])
    self.output[1] = class_scores_float:index(1, idx):typeAs(self.output[1])
    
    --self.output[9] = pairity_scores_float:index(1, idx):typeAs(self.output[9])-------!!!!!
    
    --local idx_ = torch.gt(pairity_scores_float,0):nonzero()
    
    local idx_ = torch.LongTensor(0)--idx:size(1)*(idx:size(1)-1)):zero()
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
    
    self.output[5] = lm_output_float:index(1, idx_ ):typeAs(self.output[5])-----------!!!!!!!!!!!!!!!!!!!!!!!!!!!
    --self.output[5] = lm_output_float:typeAs(self.output[5])
    

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
  --local pairity_scores = output[9]-----!!!!!!
  local captions = output[5]
  local captions = self.nets.language_model:decodeSequence(captions)
  return final_boxes, objectness_scores, captions--, pairity_scores------------------!!!!!!
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
    dbg()
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
  local D_params, grad_D_params = self.nets.discriminator_model:getParameters()--- discriminator
  return params, grad_params, cnn_params, grad_cnn_params, D_params, grad_D_params
end


function DenseCapModel:clearState()
  self.net:clearState()
  for k, v in pairs(self.crits) do
    if v.clearState then
      v:clearState()
    end
  end
end


function DenseCapModel:sample_gumbel(shape,eps)
    -----Sample from Gumbel(0,1)-------
    local U = torch.CudaTensor(shape) 
    U:uniform(0,1)
    --dbg()
    return -((-(U+eps):log()+eps):log())
end

function DenseCapModel:gumbel_softmax(logits,temp)
    --draw a sample from the Gumbel-Softmax dist -----
    local y = (logits+1e-20):log() + self:sample_gumbel(logits:size() , 1e-20)-----------for the gap of real prb and gumbel
    --y=torch.exp(y/temp)
    --sum = y:sum(2)
    --return nn.SoftMax():forward(nn.utils.recursiveType(y/temp,'torch.FloatTensor'))--softmax

    return nn.SoftMax():forward( (y/temp):float()):cuda()--
end


function DenseCapModel:generate_gumbel(gt_labels,L)
  local real = self.nets.language_model:getTarget(gt_labels)-- collect real data(caption)
  
  --real[{{}, 1}]:fill(self.START_TOKEN) --Fill the first word with <start>
  local mask = torch.eq(real, 0)
  real[mask] = real:max()-- fill empty word with <start>
    
  B,T = real:size(1), real:size(2)
  
  local Real = real.new(B*T,L+1):zero()------------------ one_hot vector
  Real=Real:scatter(2,(real+1):view(-1,1),1):reshape(B,T,L+1)
  Real = Real[{{},{},{2,-1}}]------------------------------------ trick to deal with class0, cutoff 0 class
  
  --gumbel_sentence = self:gumbel_softmax( Real:reshape(B*T,L) ,3.0)-- ----- time/memory consuming
  
  --return gumbel_sentence:reshape(B,T,L)
  return Real:reshape(B,T,L) --code for testing
end

function DenseCapModel:forward_backward_D(data)
  ------------------------------------------------------------------------------
  --           training for D
  ------------------------------------------------------------------------------  
  self:training()
  -- Run the model forward
  self:setGroundTruth(data.gt_boxes, data.gt_labels)
  local out = self:forward(data.image)
  -- Pick out the outputs we care about
  local lm_output = out[5]
  local gt_boxes = out[6]
  local gt_labels = out[7]
  local feats = out[8]
  --local pairity_scores = out[9]
  local gt_pairs = out[9]
  
  
  if lm_output == 0 then--- if no matched points, return 0
    captioning_loss = 0
    matching_loss = 0
    --dbg()
    grad_lm_output = 0--torch.CudaLongTensor(0)
  else
    
    local B, T, L= lm_output:size(1), lm_output:size(2), lm_output:size(3)
    
    local targets = torch.CudaTensor(B*2)------------------targets for GAN !!!! (B or B*T)
    targets[{{1,B}}]:fill(0)------------------fake
    targets[{{B+1 , 2*B}}]:fill(1)  --real 
    
    local gumbel_sentence = self:generate_gumbel(gt_labels,L)----real distribution
    
    local inputs = lm_output.new(B*2 , T , L):fill(0)--------------------total input for real and fake
    inputs[{{1,B},{},{}}] = lm_output:clone()--clone generated data
    inputs[{{B+1 , 2*B},{},{}}] = gumbel_sentence:clone()
    
    --dbg()
    
    local D_output = self.nets.discriminator_model:forward({inputs,feats[1]:repeatTensor(2,1)})-- sigmoid output
    local captioning_loss = self.crits.lm_crit:forward(D_output[1], targets)-- sigmoid loss func
    captioning_loss = captioning_loss * self.opt.captioning_weight
    
    local grad_dm_output = self.crits.lm_crit:backward(D_output[1], targets)
    self.nets.discriminator_model:backward( {inputs,feats[1]:repeatTensor(2,1)} , {grad_dm_output , D_output[2]:zero()} )
  end
  
    
  
  ---we need to update recognet?
  local losses = {captioning_loss=captioning_loss,}
  
  local total_loss = 0
  for k, v in pairs(losses) do
    total_loss = total_loss + v
  end
  losses.total_loss = total_loss

  return losses
end

function DenseCapModel:forward_backward_G(data)
  ------------------------------------------------------------------------------
  --           training for G
  ------------------------------------------------------------------------------  
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
  local feats = out[8]  
  --local pairity_scores = out[9]
  local gt_pairs = out[9]

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
  local captioning_loss = 0
  local matching_loss = 0
  if lm_output == 0 then--- if no matched points, return 0
    captioning_loss = 0
    matching_loss = 0
    --dbg()
    grad_lm_output = 0--torch.CudaLongTensor(0)
  else
    local B, T, L= lm_output:size(1), lm_output:size(2), lm_output:size(3)
    
    -- Compute captioning loss  
    local D_output = self.nets.discriminator_model:forward({ lm_output,feats[1]})--- forward D
    
    local target = D_output[1].new(D_output[1]:size()):fill(1)------------------ label for GAN !!!!
    --local target = torch.CudaTensor(B):fill(1)------------------targets for GAN !!!! (B or B*T)
    
    
    local gumbel_sentence = self:generate_gumbel(gt_labels,L)-- collect real data(caption)
    local real_output = self.nets.discriminator_model:forward({gumbel_sentence,feats[1]})---real embedding
  --dbg()
  
    captioning_loss = self.crits.lm_crit:forward(D_output[1] , target)
    matching_loss = self.crits.matching_crit:forward(D_output[2], real_output[2])
    
    captioning_loss = captioning_loss * self.opt.captioning_weight
    matching_loss = matching_loss * self.opt.captioning_weight  
    
    local grad_dm_output1 = self.crits.lm_crit:backward(D_output[1] , target)
    local grad_dm_output2 = self.crits.matching_crit:backward(D_output[2], real_output[2])
    grad_lm_output = self.nets.discriminator_model:backward({lm_output,feats[1]}, {grad_dm_output1,grad_dm_output2})
    grad_lm_output[1]:mul(self.opt.captioning_weight)
  end

  local ll_losses = self.nets.localization_layer.stats.losses
  local losses = {
    mid_objectness_loss=ll_losses.obj_loss_pos + ll_losses.obj_loss_neg,
    mid_box_reg_loss=ll_losses.box_reg_loss,
    end_objectness_loss=end_objectness_loss,
    --pairity_loss = pairity_loss,-----------------------------!!!!!!!!!!!!!!!
    end_box_reg_loss=end_box_reg_loss,
    captioning_loss=captioning_loss,
    matching_loss = matching_loss,
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
  grad_out[5] = grad_lm_output[1]
  grad_out[6] = gt_boxes.new(#gt_boxes):zero()
  grad_out[7] = gt_labels.new(#gt_labels):zero()
  
  grad_out[8] = {out[8][1].new(#out[8][1]):zero(),out[8][2].new(#out[8][2]):zero()}-------------!!!!!!!!!!!!!!!!!!!!!!!!!
  --grad_out[9] = grad_pairity_scores-----------!!!!!!!!!!!!
  --
  grad_out[9] = gt_pairs.new((#gt_pairs)[1],1):zero()
  --dbg()
  self:backward(input, grad_out)

  return losses
end





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
  --local pairity_scores = out[9]
  local gt_pairs = out[9]

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
  if lm_output == 0 then--- if no matched points, return 0
    captioning_loss = 0
    --dbg()
    grad_lm_output = 0--torch.CudaLongTensor(0)
  else
  
    local target = self.nets.language_model:getTarget(gt_labels)
    captioning_loss = self.crits.lm_crit:forward(lm_output, target)
    captioning_loss = captioning_loss * self.opt.captioning_weight
    grad_lm_output = self.crits.lm_crit:backward(lm_output, target)
    grad_lm_output:mul(self.opt.captioning_weight)
  end

  local ll_losses = self.nets.localization_layer.stats.losses
  local losses = {
    mid_objectness_loss=ll_losses.obj_loss_pos + ll_losses.obj_loss_neg,
    mid_box_reg_loss=ll_losses.box_reg_loss,
    end_objectness_loss=end_objectness_loss,
    --pairity_loss = pairity_loss,-----------------------------!!!!!!!!!!!!!!!
    end_box_reg_loss=end_box_reg_loss,
    captioning_loss=captioning_loss,
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
  grad_out[5] = grad_lm_output
  grad_out[6] = gt_boxes.new(#gt_boxes):zero()
  grad_out[7] = gt_labels.new(#gt_labels):zero()
  grad_out[8] = out[8].new(#out[8]):zero()-------------!!!!!!!!!!!!!!!!!!!!!!!!!
  --grad_out[9] = grad_pairity_scores-----------!!!!!!!!!!!!
  grad_out[9] = gt_pairs.new((#gt_pairs)[1],1):zero()
  
  self:backward(input, grad_out)

  return losses
end
