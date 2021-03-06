local dbg = require("debugger")
require 'nn'
require 'torch-rnn'

local utils = require 'densecap.utils'

local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')


function LM:__init(opt)
  parent.__init(self)

  opt = opt or {}
  self.vocab_size = utils.getopt(opt, 'vocab_size')
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.image_vector_dim = utils.getopt(opt, 'image_vector_dim')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.seq_length = utils.getopt(opt, 'seq_length') --!!!!!!!!!!!!!
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  self.idx_to_token = utils.getopt(opt, 'idx_to_token')
  self.dropout = utils.getopt(opt, 'dropout', 0)
  
  local W, D = self.input_encoding_size, self.image_vector_dim
  local V, H = self.vocab_size, self.rnn_size
  local S = 0 --- size of spatial feature----------------------------!!!!!!!!!!!!!!! S = 0/10/64
  local num_input = 3
  
  -- For mapping from image vectors to word vectors
  self.image_encoder = nn.Sequential()  
  
  
  self.image_encoder:add(nn.SpatialAveragePooling(7,7))------!!!!2222(B, 512,7,7)->(B, 512,1,1)
  self.image_encoder:add(nn.View(-1):setNumInputDims(3))--!!!222 (B, 512,1,1) -> (B,512)
  
  --self.image_encoder:add(nn.View(-1):setNumInputDims(3))
  --self.image_encoder:add(nn.Linear(25088,D))
  --self.image_encoder:add(nn.ReLU(true))
  --self.image_encoder:add(nn.Dropout(0.5))
  --self.image_encoder:add(nn.Linear(D,D))
  --self.image_encoder:add(nn.ReLU(true))
  --self.image_encoder:add(nn.Dropout(0.5))
  
  self.image_encoder:add(nn.Linear(W, W))----!!!222 (B,4096)-> (B,512)
  self.image_encoder:add(nn.ReLU(true))
  ----------------------------------------
  
  if not(num_input==1) then
    self.image_encoder2 = nn.Sequential() 
     
    self.image_encoder2:add(nn.SpatialAveragePooling(7,7))------!!!!2222(B, 512,7,7)->(B, 512,1,1)
    self.image_encoder2:add(nn.View(-1):setNumInputDims(3))--!!!222 (B, 512,1,1) -> (B,512)
    
    self.image_encoder2:add(nn.Linear(W, W))----!!!222 (B,4096)-> (B,512)
    self.image_encoder2:add(nn.ReLU(true))
    -----------------------------------------
    self.image_encoder3 = nn.Sequential()  
    
    self.image_encoder3:add(nn.SpatialAveragePooling(7,7))------!!!!2222(B, 512,7,7)->(B, 512,1,1)
    self.image_encoder3:add(nn.View(-1):setNumInputDims(3))--!!!222 (B, 512,1,1) -> (B,512)
    
    self.image_encoder3:add(nn.Linear(W, W))----!!!222 (B,4096)-> (B,512)
    self.image_encoder3:add(nn.ReLU(true))
  end

  
  self.START_TOKEN = self.vocab_size + 1
  self.END_TOKEN = self.vocab_size + 1
  self.NULL_TOKEN = self.vocab_size + 2

  -- For mapping word indices to word vectors
  local V, W = self.vocab_size, self.input_encoding_size
  self.lookup_table = nn.LookupTable(V + 2, W)----!!! 
  
  -- Change this to sample from the distribution instead
  self.sample_argmax = true

  
  
  
  
  
  self.image_encoder_real = nn.Sequential()
  
  if num_input==1 then
    self.image_encoder:add(self.image_encoder)
  elseif num_input==3 then
    local combine = nn.ParallelTable()---merge imvec and mask activation
    combine:add(self.image_encoder)
    combine:add(self.image_encoder2)
    combine:add(self.image_encoder3)
    self.image_encoder_real:add(combine)
    self.image_encoder_real:add(nn.JoinTable(2, 2))----( concat,sum,mul)
    self.image_encoder_real:add(nn.Linear(W*3, W))
    self.image_encoder_real:add(nn.ReLU(true))
    self.image_encoder_real:add(nn.Dropout(self.dropout))
    --self.image_encoder_real:add(nn.Linear(256, W))
    --self.image_encoder_real:add(nn.ReLU(true))
  else
    dbg()
  end
  
 
  self.image_encoder_real:add(nn.View(1, -1):setNumInputDims(1))
  ------------------------------------------------------------------------------
  -- self.rnn maps wordvecs of shape N x T x W to word probabilities
  -- of shape N x T x (V + 1)
   self.rnn = nn.Sequential()
  for i = 1, self.num_layers do
    local input_dim = self.rnn_size
    if i == 1 then
      input_dim = self.input_encoding_size
    end
    self.rnn:add(nn.LSTM(input_dim+S, self.rnn_size))----!!!! 64
    if self.dropout > 0 then
      self.rnn:add(nn.Dropout(self.dropout))
    end
  end

  self.view_in = nn.View(1, 1, -1):setNumInputDims(3)
  self.view_out = nn.View(1, -1):setNumInputDims(2)
  self.rnn:add(self.view_in)
  
  
  ----------------------------------------
  self.rnn:add(nn.Linear(H, V + 1))
  --self.rnn:add(nn.SoftMax())-----------------------------------------------------------!!!!!!!!!!!!!!!!!!
  self.rnn:add(self.view_out)
  
  
  
  -- self.net maps a table {image_vecs, gt_seq} to word probabilities
  self.net = nn.Sequential()
  local parallel = nn.ParallelTable()
  parallel:add(self.image_encoder_real)-------!!!
  parallel:add(self.start_token_generator)
  parallel:add(self.lookup_table)
  self.net:add(parallel)
  self.net:add(nn.JoinTable(1, 2))
  self.net:add(self.rnn)
  
  
  
  self:training()
end


--[[
Decodes a sequence into a table of strings

Inputs:
- seq: tensor of shape N x T

Returns:
- captions: Array of N strings
--]]
function LM:decodeSequence(seq)
  local delimiter = ' '
  local captions = {}
  local N, T = seq:size(1), seq:size(2)
  for i = 1, N do
    local caption = ''
    for t = 1, T do
      local idx = seq[{i, t}]
      if idx == self.END_TOKEN or idx == 0 then break end
      if t > 1 then
        caption = caption .. delimiter
      end
      caption = caption .. self.idx_to_token[idx]
    end
    table.insert(captions, caption)
  end
  return captions
end


function LM:updateOutput(input)
  self.recompute_backward = true
  local union_vectors = input[1]
  local gt_sequence = input[2]
  local Masks = input[3]:cuda() 
  
  if ##union_vectors == 0  then--- if no matched pairs, don't compute loss !!
    self._forward_sampled = false
    return 0
  end
  
  local subj_vectors = input[4][1]--!!!
  local obj_vectors = input[4][2]--!!!
  
  if gt_sequence:nElement() > 0 then  
    
    -- Add a start token to the start of the gt_sequence, and replace
    -- 0 with NULL_TOKEN
    local N, T = gt_sequence:size(1), gt_sequence:size(2)
    self._gt_with_start = gt_sequence.new(N, T + 1)
    self._gt_with_start[{{}, 1}]:fill(self.START_TOKEN)
    self._gt_with_start[{{}, {2, T + 1}}]:copy(gt_sequence)
    local mask = torch.eq(self._gt_with_start, 0)
    self._gt_with_start[mask] = self.NULL_TOKEN
    
    -- Reset the views around the nn.Linear
    self.view_in:resetSize(N * (T + 2), -1)
    self.view_out:resetSize(N, T + 2, -1)
    
    
    --dbg()
    if ##Masks==0 then
      --dbg()
      
      if ##obj_vectors==0 then
        self.output = self.net:updateOutput{union_vectors, self._gt_with_start}
      else
        --dbg()
        self.output = self.net:updateOutput{{union_vectors,subj_vectors,obj_vectors}, self._gt_with_start}
      end
    else
      self.output = self.net:updateOutput{}----!!!!3333
    end
    --dbg()
    self._forward_sampled = false
    
    return self.output
  else
    self._forward_sampled = true
    if self.beam_size ~= nil then
      print 'running beam search'
      self.output = self:beamsearch(image_vectors, self.beam_size)
      return self.output
    else

      return self:sample(union_vectors, subj_vectors, obj_vectors, Masks)---!!!!!

    end
  end
end


--[[
Convert a ground-truth sequence of shape to a target suitable for the
TemporalCrossEntropyCriterion from torch-rnn.

Input:
- gt_sequence: Tensor of shape (N, T) where each element is in the range [0, V];
  an entry of 0 is a null token.
--]]
function LM:getTarget(gt_sequence)
  -- Make sure it's on CPU since we will loop over it
  local gt_sequence_long = gt_sequence:long()
  local N, T = gt_sequence:size(1), gt_sequence:size(2)
  local target = torch.LongTensor(N, T + 2):zero()
  target[{{}, {2, T + 1}}]:copy(gt_sequence)
  for i = 1, N do
    for t = 2, T + 2 do
      if target[{i, t}] == 0 then
        -- Replace the first null with an end token
        target[{i, t}] = self.END_TOKEN
        break
      end
    end
  end
  return target:type(gt_sequence:type())
end


function LM:sample(union_vectors, subj_vectors, obj_vectors,Masks)
  local N, T = union_vectors:size(1), self.seq_length
  local seq = torch.LongTensor(N, T):zero()
  local softmax = nn.SoftMax():type(union_vectors:type())
  -- During sampling we want our LSTM modules to remember states
  for i = 1, #self.rnn do
    local layer = self.rnn:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = true
    end
  end
  --dbg()
  -- Reset view sizes
  self.view_in:resetSize(N, -1)
  self.view_out:resetSize(N, 1, -1)
  

  -- First timestep: image vectors, ignore output
  
  local union_vecs_encoded
  if not(##obj_vectors==0) then
    local subj_vecs_encoded
    local obj_vecs_encoded
  end
  
  if ##Masks==0 then
    if ##obj_vectors==0 then
      vecs_encoded  = self.image_encoder_real:forward(union_vectors)
    else
      vecs_encoded  = self.image_encoder_real:forward{union_vectors,subj_vectors,obj_vectors}
    end
    
  else
    image_vecs_encoded = self.image_encoder:forward(image_vectors)---!!!!333
    spatial_vecs_encoded = self.image_encoder2:forward(Masks)---!!!!333
  end
  
  --dbg()
  self.rnn:forward(vecs_encoded)---!!!333


  -- Now feed words through RNN
  for t = 1, T do
    local words = nil
    if t == 1 then
      -- On the first timestep, feed START tokens
      words = torch.LongTensor(N, 1):fill(self.START_TOKEN)
    else
      -- On subsequent timesteps, feed previously sampled words
      words = seq[{{}, {t-1, t-1}}]
    end
    local wordvecs = self.lookup_table:forward(words)
    
    
    local scores = self.rnn:forward(wordvecs):view(N, -1)


    local idx = nil
    if self.sample_argmax then
      _, idx = torch.max(scores, 2)
    else
      local probs = softmax:forward(scores)
      idx = torch.multinomial(probs, 1):view(-1):long()
    end
    seq[{{}, t}]:copy(idx)
  end

  -- After sampling stop remembering states
  for i = 1, #self.rnn do
    local layer = self.rnn:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = false
    end
  end

  self.output = seq
  return self.output
end


function LM:updateGradInput(input, gradOutput)
  if self.recompute_backward then
    self:backward(input, gradOutput)
  end
  return self.gradInput
end


function LM:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end

function LM:backward(input, gradOutput, scale)
  assert(self._forward_sampled == false, 'cannot backprop through sampling')
  assert(scale == nil or scale == 1.0)
  self.recompute_backward = false

  local net_input
  
    
--dbg()
  if gradOutput == 0 then---if there is no box- pair detected!
    if ##input[3]==0 then
      self.gradInput = {input[1]:new():zero(),nil,input[3]:new():zero(), input[4]:new():zero()}
    else
      dbg()
      self.gradInput = { input[1]:new():zero() ,nil,  input[2]:new():zero() ,{input[4][1]:new():zero(),input[4][2]:new():zero()}}
    end
  else
    if ##input[3]==0 then ---Mask !!!!!
      --if not(#input[4]==2) then
      --  dbg()
      --end
      net_input = { {input[1]:cuda(), input[4][1]:cuda(), input[4][2]:cuda()}, self._gt_with_start }
    else
      dbg()
      net_input = {}
    end
  
     local tmp = self.net:backward(net_input, gradOutput, scale)
     if ##input[3]==0 then--!!!!!
       self.gradInput = {tmp[1][1],nil,nil,{tmp[1][2],tmp[1][3]} }
     else
       dbg()
       self.gradInput = {tmp[1][1],nil , tmp[1][2],{tmp[2][1],tmp[3][1]}}---!!!777
     end
  end

  self.gradInput[2] = input[2]:new():zero()
  return self.gradInput
end



function LM:parameters()
  return self.net:parameters()
end


function LM:training()
  parent.training(self)
  self.net:training()
end


function LM:evaluate()
  parent.evaluate(self)
  self.net:evaluate()
end


function LM:clearState()
  self.net:clearState()
end


function LM:parameters()
  return self.net:parameters()
end

