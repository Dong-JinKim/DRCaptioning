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
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  self.idx_to_token = utils.getopt(opt, 'idx_to_token')
  self.dropout = utils.getopt(opt, 'dropout', 0)

  local W, D = self.input_encoding_size, self.image_vector_dim
  local V, H = self.vocab_size, self.rnn_size

  -- For mapping from image vectors to word vectors
  self.image_encoder = nn.Sequential()
  self.image_encoder:add(nn.Linear(2*D, W))------------------!!!!!
  self.image_encoder:add(nn.ReLU(true))
  self.image_encoder:add(nn.View(1, -1):setNumInputDims(1))
  
  self.START_TOKEN = self.vocab_size + 1
  self.END_TOKEN = self.vocab_size + 1
  self.NULL_TOKEN = self.vocab_size + 2

  -- For mapping word indices to word vectors
  local V, W = self.vocab_size, self.input_encoding_size
  self.lookup_table = nn.LookupTable(V + 2, W)
  
  -- Change this to sample from the distribution instead
  self.sample_argmax = true

  -- self.rnn maps wordvecs of shape N x T x W to word probabilities
  -- of shape N x T x (V + 1)
  self.rnn = nn.Sequential()
  for i = 1, self.num_layers do
    local input_dim = self.rnn_size
    if i == 1 then
      input_dim = self.input_encoding_size
    end
    self.rnn:add(nn.LSTM(input_dim, self.rnn_size))
    if self.dropout > 0 then
      self.rnn:add(nn.Dropout(self.dropout))
    end
  end

  self.view_in = nn.View(1, 1, -1):setNumInputDims(3)
  self.view_out = nn.View(1, -1):setNumInputDims(2)
  self.rnn:add(self.view_in)
  self.rnn:add(nn.Linear(H, V + 1))
  --self.rnn:add(nn.SoftMax())-----------------------------------------------------------!!!!!
  self.rnn:add(self.view_out)
  

  -- self.net maps a table {image_vecs, gt_seq} to word probabilities
  self.net = nn.Sequential()
  local parallel = nn.ParallelTable()
  parallel:add(self.image_encoder)
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
  local image_vectors = input[1][1]----!!!!!
  local gt_sequence = input[2]
  local label_idx = input[1][2]-----!!!!!!
  --dbg()
  self.label_idx = label_idx
  
  if ##self.label_idx == 0 then--- if no matched pairs, don't compute loss
    self._forward_sampled = false
    return 0
  end
  
  if gt_sequence:nElement() > 0 then
    return self:forward_LM(image_vectors)
  else
      if self.beam_size ~= nil then
        print 'running beam search'
        self.output = self:beamsearch(image_vectors, self.beam_size)
        return self.output
      else
        return self:sample(image_vectors)
      end
  end
  
end


function LM:forward_LM(image_vectors)
  
  local N, T, L  = image_vectors:size(1), self.seq_length, self.vocab_size
  self._gt_with_start = torch.LongTensor(N,T+1 ):zero()
  self._gt_with_start[{{}, 1}]:fill(self.START_TOKEN)--- the first word is <start>
  
  local output = torch.CudaTensor(N, T+2, L+1):zero()
  --output[{{},1, self.START_TOKEN}]:fill(1)--- the first word is <start>
  
  local Null_mask = torch.CudaByteTensor(N,1):zero() -- if we met <END>, then end forever
  
  local softmax = nn.SoftMax():type(image_vectors:type())
  -- During sampling we want our LSTM modules to remember states
  for i = 1, #self.rnn do
    local layer = self.rnn:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = true
    end
  end

  -- Reset view sizes
  self.view_in:resetSize(N, -1)
  self.view_out:resetSize(N, 1, -1)
  
  -- First timestep: image vectors, ignore output
  local image_vecs_encoded = self.image_encoder:forward(image_vectors)
  output[{ {},1,{}  }] = self.rnn:forward(image_vecs_encoded)

  -- Now feed words through RNN
  for t = 1, T+1 do
    local words = nil

    -- On subsequent timesteps, feed previously sampled words
    words = self._gt_with_start[{{}, {t, t}}]
    
    --dbg()
    local wordvecs = self.lookup_table:forward(words)
    local scores = self.rnn:forward(wordvecs):view(N, -1)
    local idx = nil
    --dbg()
    if self.sample_argmax then
      _, idx = torch.max(scores, 2)
      output[{ {},t+1,{}  }]:copy(scores)
    else
      local probs = softmax:forward(scores)
      idx = torch.multinomial(probs, 1):long()--!!! view(-1)
      output[{ {},t+1,{}  }]:copy(probs)
    end
    
    if t < T+1 then--collect predicted words
        Null_mask = Null_mask +  torch.eq(idx,self.END_TOKEN)
        idx[Null_mask] = self.NULL_TOKEN---if we met End,fill the GT from here <Null>
        self._gt_with_start[{{}, t+1}]:copy( idx  )
    end
  end

  -- After sampling stop remembering states
  for i = 1, #self.rnn do
    local layer = self.rnn:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = false
    end
  end

  self.view_in:resetSize(N * (T+2), -1)
  self.view_out:resetSize(N, T+2 , -1)

  --self.output = output
  self.output = self.net:forward({image_vectors,self._gt_with_start})
  self._forward_sampled = false
  
  return self.output
end




--[[
Convert a ground-truth sequence of shape to a target suitable for the
TemporalCrossEntropyCriterion from torch-rnn.

Input:
- gt_sequence: Tensor of shape (N, T) where each element is in the range [0, V];
  an entry of 0 is a null token.
--]]
function LM:getTarget(gt_sequence)
  
  gt_sequence = gt_sequence:index(1,self.label_idx)---!!!
  
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

function LM:sample(image_vectors)
  dbg()
  local N, T = image_vectors:size(1), self.seq_length
  local seq = torch.LongTensor(N, T):zero()
  local softmax = nn.SoftMax():type(image_vectors:type())
  
  -- During sampling we want our LSTM modules to remember states
  for i = 1, #self.rnn do
    local layer = self.rnn:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = true
    end
  end

  -- Reset view sizes
  self.view_in:resetSize(N, -1)
  self.view_out:resetSize(N, 1, -1)

  -- First timestep: image vectors, ignore output
  local image_vecs_encoded = self.image_encoder:forward(image_vectors)
  self.rnn:forward(image_vecs_encoded)
  

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
  local net_input = {input[1][1], self._gt_with_start}-----------------------------!!!
  if gradOutput == 0  then---if there is no box- pair detected!
    self.gradInput = {torch.CudaLongTensor(8,8192):zero(),torch.LongTensor(8):zero()}--input[1].new(#input[1]):zero()
  else
    --dbg()
    self.gradInput = self.net:backward(net_input, gradOutput, scale)--!!!!
    
  end
  -- dbg()
  self.gradInput[2] = input[2].new(#input[2]):zero()
  --dbg()
  self.gradInput[1] = {self.gradInput[1], input[1][2].new(input[1][2]:size()[1]):zero()}---!!!!!
  --dbg()
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

--[[
image_vectors: N x D
--]]
function LM:beamsearch(image_vectors, beam_size)
  error('Not implemented')
end
