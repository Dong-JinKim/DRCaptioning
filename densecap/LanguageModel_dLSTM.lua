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
  self.seq_length = utils.getopt(opt, 'seq_length')/2 --!!!!!!!!!!!!!
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  self.idx_to_token = utils.getopt(opt, 'idx_to_token')
  self.dropout = utils.getopt(opt, 'dropout', 0)
  
  local W, D = self.input_encoding_size, self.image_vector_dim
  local V, H = self.vocab_size, self.rnn_size
  local S = 6 --- size of spatial feature----------------------------!!!!!!!!!!!!!!! S = 0/10/64

  -- For mapping from image vectors to word vectors
  self.image_encoder = nn.Sequential()
  
  
  ---!!!!!!!!!!!!!!!!!!!!! if you want 1*1conv first
  --self.image_encoder:add(nn.SpatialConvolution(512*3, 256, 1, 1))---!!!!555
  --self.image_encoder:add(nn.SpatialConvolution(256, W, 1, 1))
  --self.image_encoder:add(nn.ReLU(true))
  ------------------------------------------
  
  self.image_encoder:add(nn.SpatialAveragePooling(7,7))------!!!!2222(B, 512,7,7)->(B, 512,1,1)
  self.image_encoder:add(nn.View(-1):setNumInputDims(3))--!!!222 (B, 512,1,1) -> (B,512)
  
  self.image_encoder:add(nn.Linear(512, W))----!!!222
  --self.image_encoder:add(nn.Linear(512*3, 128))----!!!222
  --self.image_encoder:add(nn.Linear(128, W))---!!!!!!555
  self.image_encoder:add(nn.ReLU(true))
  
  
  
  
  
  self.image_encoder2 = nn.Sequential()
  if S == 10 then
    self.image_encoder2:add(nn.Identity())
    
  elseif S==6 then
    self.image_encoder2:add(nn.Linear(6,64))---!!!!
    --self.image_encoder2:add(nn.Dropout(self.dropout))
  else
    dbg()
  end

  
  
  self.image_encoder:add(nn.View(1, -1):setNumInputDims(1)) 
  self.image_encoder2:add(nn.View(1, -1):setNumInputDims(1))
  
  self.START_TOKEN = self.vocab_size + 1
  self.END_TOKEN = self.vocab_size + 1
  self.NULL_TOKEN = self.vocab_size + 2

  -- For mapping word indices to word vectors
  local V, W = self.vocab_size, self.input_encoding_size
  self.lookup_table = nn.LookupTable(V + 2, W)----!!! 64
  self.lookup_table2 = nn.LookupTable(V + 2, 64)----!!! 64
  
  -- Change this to sample from the distribution instead
  self.sample_argmax = true

  
  ------------------------------------------------------------------------------
  -- self.rnn maps wordvecs of shape N x T x W to word probabilities
  -- of shape N x T x (V + 1)
  self.rnn1 = nn.Sequential()---------LSTM1
  for i = 1, self.num_layers do
    local input_dim = self.rnn_size
    if i == 1 then
      input_dim = self.input_encoding_size
    end
    self.rnn1:add(nn.LSTM(input_dim, self.rnn_size))
    if self.dropout > 0 then
      self.rnn1:add(nn.Dropout(self.dropout))
    end
  end
  local parallel1 = nn.ParallelTable()
  parallel1:add(self.image_encoder)
  parallel1:add(self.start_token_generator)
  parallel1:add(self.lookup_table)
  self.input1 = nn.Sequential()
  self.input1:add(parallel1)
  self.input1:add(nn.JoinTable(1, 2))
  self.input1:add(self.rnn1)
  ------------------------------------------------------------------------------
  self.rnn2 = nn.Sequential()---------LSTM2
  for i = 1, self.num_layers do
    local input_dim = self.rnn_size
    if i == 1 then
      input_dim = self.input_encoding_size
    end
    self.rnn2:add(nn.LSTM(64, self.rnn_size))
    if self.dropout > 0 then
      self.rnn2:add(nn.Dropout(self.dropout))
    end
  end
  local parallel2 = nn.ParallelTable()
  parallel2:add(self.image_encoder2)
  parallel2:add(self.start_token_generator)
  parallel2:add(self.lookup_table2)
  self.input2 = nn.Sequential()
  self.input2:add(parallel2)
  self.input2:add(nn.JoinTable(1, 2))
  self.input2:add(self.rnn2)
  ------------------------------------------------------------------------------
  self.view_in = nn.View(1, 1, -1):setNumInputDims(3)
  self.view_out = nn.View(1, -1):setNumInputDims(2)

  self.out=nn.Sequential()
  self.out:add(self.view_in)
  self.out:add(nn.Linear(H*2, H))
  self.out:add(nn.Linear(H, V + 1))
  self.out:add(self.view_out)
  
  local combine = nn.ParallelTable()---merge imvec and mask activation
  combine:add(self.input1)
  combine:add(self.input2)
  
  -- self.net maps a table {image_vecs, gt_seq} to word probabilities
  self.net = nn.Sequential()
  self.net:add(combine)
  self.net:add(nn.JoinTable(2, 2))
  self.net:add(self.out)
  
  
  
  
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
  local image_vectors = input[1]
  local gt_sequence = input[2]
  local Masks = input[3]:cuda() ------!!!!!222
  if ##image_vectors == 0  then--- if no matched pairs, don't compute loss !!
    self._forward_sampled = false
    return 0
  end
  
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
      
      self.output = self.net:updateOutput{image_vectors, self._gt_with_start}
    else
      self.output = self.net:updateOutput{{image_vectors,self._gt_with_start},{Masks, self._gt_with_start}}----!!!!3333
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
      --if Mask==nil then
      --  return self:sample(image_vectors)
      --else
      return self:sample(image_vectors,Masks)---!!!!!
      --end
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


function LM:sample(image_vectors,Masks)
  local N, T = image_vectors:size(1), self.seq_length
  local seq = torch.LongTensor(N, T):zero()
  local softmax = nn.SoftMax():type(image_vectors:type())
  -- During sampling we want our LSTM modules to remember states
  for i = 1, #self.rnn1 do
    local layer = self.rnn1:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = true
    end
  end
  for i = 1, #self.rnn2 do---!!!333
    local layer = self.rnn2:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = true
    end
  end

  -- Reset view sizes
  self.view_in:resetSize(N, -1)
  self.view_out:resetSize(N, 1, -1)
  

  -- First timestep: image vectors, ignore output
  
  local image_vecs_encoded--!!!!!!!!!!222
  local spatial_vecs_encoded
  if ##Masks==0 then
    image_vecs_encoded  = self.image_encoder:forward(image_vectors)
  else
    image_vecs_encoded = self.image_encoder:forward(image_vectors)---!!!!333
    spatial_vecs_encoded = self.image_encoder2:forward(Masks)---!!!!333
  end
  
  
  self.rnn1:forward(image_vecs_encoded)---!!!333
  self.rnn2:forward(spatial_vecs_encoded)---!!!333

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
    local wordvecs2 = self.lookup_table2:forward(words)
    
    local scores = self.out:forward(  torch.cat( self.rnn1:forward(wordvecs), self.rnn2:forward(wordvecs2) ,3)    ):view(N, -1)


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
  for i = 1, #self.rnn1 do
    local layer = self.rnn1:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = false
    end
  end
  -- After sampling stop remembering states
  for i = 1, #self.rnn2 do
    local layer = self.rnn2:get(i)
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
  if ##input[3]==0 then ---Mask !!!!!
    net_input = {input[1], self._gt_with_start}
  else
    net_input = { {input[1]:cuda(), self._gt_with_start},{input[3]:cuda(), self._gt_with_start} }
  end
    
--dbg()
  if gradOutput == 0 then---if there is no box- pair detected!
    if ##input[3]==0 then
      self.gradInput = {input[1].new(0):zero()}
    else
      self.gradInput = { input[1].new(input[1]):zero() ,nil,  input[2].new(input[2]):zero() }
    end
  else
     local tmp = self.net:backward(net_input, gradOutput, scale)
     if ##input[3]==0 then--!!!!!
       self.gradInput = tmp
     else
       self.gradInput = {tmp[1][1],nil, tmp[2][1]}---!!!777
     end
  end

  self.gradInput[2] = input[2].new(#input[2]):zero()
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

