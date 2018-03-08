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
  
  self.ATT = true

  local W, D = self.input_encoding_size, self.image_vector_dim
  local V, H = self.vocab_size, self.rnn_size
  local S = 0 --- size of spatial feature----------------------------!!!!!!!!!!!!!!!
  
  --self.att_input = 
  
  -- For mapping from image vectors to word vectors
  
  
  self.image_encoder = nn.Sequential()
  --self.image_encoder:add(nn.SpatialAveragePooling(7,7))------!!!!2222
  --self.image_encoder:add(nn.View(-1,512))--!!!222
  --self.image_encoder:add(nn.Linear(512, W))----!!!222
  --self.image_encoder:add(nn.ReLU(true))
  

  if S==0 then
    if self.ATT then-- for attention module !!!444
      self.image_encoder:add(nn.Replicate(self.seq_length+1,2))---!!!!!!!3333 (B,512,7,7)->(B,T,512,7,7)
      self.image_encoder:add(nn.Contiguous())--!!!4
      self.image_encoder:add(nn.View(512,-1):setNumInputDims(3)) -- (B,T,512,7,7)->(B*T,512,49)    
      
      ----!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      --self.image_encoder:add(nn.Transpose({2,3}):setNumInputDims(3))-- (B*T,512,49)->(B*T,49,512)    !!!!!555
      --self.image_encoder:add(nn.View(-1,1):setNumInputDims(1))-- (B*T,49,512)->(B*T*49,512,1) !!!!!555
      -----------------------------------------------------------------------------------------------------------
      
      self.image_encoder2 = nn.Sequential()--!!! (B,T,49)
      self.image_encoder2:add(nn.View(-1,1):setNumInputDims(1))---!!! (B,T,49)->(B*T,49,1) 
      
      ----!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      --self.image_encoder2:add(nn.View(1,1):setNumInputDims(1))---!!! (B*T,49,1)->(B*T*49,1,1) !!!!!5555
      -----------------------------------------------------------------------------------------------------------
      
      
      local image_encoder12 = nn.ParallelTable()---merge imvec and mask activation
      image_encoder12:add(self.image_encoder)
      image_encoder12:add(self.image_encoder2)
      
      self.image_encoder_real = nn.Sequential()
      self.image_encoder_real:add(image_encoder12)
      
      self.image_encoder_real:add(nn.MM())-- (B*T,D,49)*(B*T,49,1)->(B,D,1)
      --(B*T*49,512,1) * (B*T*49,1,1) -> (B*T*49,512,1)
      
      ----!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      --self.image_encoder_real:add(nn.View(-1,49,512):setNumInputDims(3))--(B*T*49,512,1)->(B*T,49,512) --!!! 555
      --self.image_encoder_real:add(nn.Transpose({2,3}):setNumInputDims(3))--(B*T,49,512)->(B*T,512,49) --!!! 555
      --self.image_encoder_real:add(nn.View(-1,7,7):setNumInputDims(2))--(B*T,512,49)->(B*T,512,7,7) --!!! 555
      --self.image_encoder_real:add( nn.SpatialConvolution(512,512,7,7) )--(B*T,512,7,7)->(B*T,512,1,1) --!!! 555
      --self.image_encoder_real:add(nn.ELU(1,true))
      --self.image_encoder_real:add(nn.View(1, -1):setNumInputDims(3))----!!!  (B*T,D,1,1)->(B*T,1,D)--!!! 555
      -----------------------------------------------------------------------------------------------------------
      self.image_encoder_real:add(nn.View(1, -1):setNumInputDims(2))----!!!  (B*T,D,1)->(B*T,1,D)--!!! 555
    
    else--baseline
      self.image_encoder:add(nn.Replicate(self.seq_length+1,2))---!!!!!!!3333 (B,512)->(B,16,512)
      --self.image_encoder:add(nn.View(1, -1):setNumInputDims(1))  --- (B,512)->(B,1,512)
    end
    
  else
    self.image_encoder2 = nn.Sequential()
    if S == 10 then
      self.image_encoder2:add(nn.Identity())
    elseif S==64 then
      self.image_encoder2:add(nn.SpatialConvolution(2, 96, 5, 5, 2, 2))
      self.image_encoder2:add(nn.ELU(1,true))
      self.image_encoder2:add(nn.SpatialConvolution(96, 128, 5, 5, 2, 2))
      self.image_encoder2:add(nn.ELU(1,true))
      self.image_encoder2:add(nn.SpatialConvolution(128, 64, 5, 5, 2, 2))
      self.image_encoder2:add(nn.ELU(1,true))
      self.image_encoder2:add(nn.View(-1):setNumInputDims(3))
    end
    
    local image_encoder12 = nn.ParallelTable()---merge imvec and mask activation
    image_encoder12:add(self.image_encoder)
    image_encoder12:add(self.image_encoder2)
    
    self.image_encoder_real = nn.Sequential()
    self.image_encoder_real:add(image_encoder12)
    self.image_encoder_real:add(nn.JoinTable(1,2))
    self.image_encoder_real:add(nn.View(1, -1):setNumInputDims(1))----!!!
  end
  
  
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
    self.rnn:add(nn.LSTM(input_dim*2+S, self.rnn_size+S))----!!!! 333  input_dim->input_dim*2
    if self.dropout > 0 then
      self.rnn:add(nn.Dropout(self.dropout))
    end
  end

  self.view_in = nn.View(1, 1, -1):setNumInputDims(3)
  self.view_out = nn.View(1, -1):setNumInputDims(2)
  self.view_out2 = nn.View(1, -1):setNumInputDims(2)--!!!
  self.rnn:add(self.view_in)

  
  
  
  
  if self.ATT then---!!!!!!!!!!!!!!!!!!!!!!44444
    local cap_branch = nn.Sequential()--branch for word output
    cap_branch:add(nn.Linear(H+S, V + 1))
    --self.rnn:add(nn.SoftMax())-----------------------------------------------------------!!!!!!!!!!!!!!!!!!
    cap_branch:add(self.view_out)
    
    local att_branch = nn.Sequential()-- branch for attention
    att_branch:add(nn.Linear(H+S, 3))-- parts class !!!!!
    att_branch:add(self.view_out2)
    
    local branch = nn.ConcatTable()
    branch:add(cap_branch)
    branch:add(att_branch)
    
    self.rnn:add(branch)
  else
    self.rnn:add(nn.Linear(H+S, V + 1))
    --self.rnn:add(nn.SoftMax())-----------------------------------------------------------!!!!!!!!!!!!!!!!!!
    self.rnn:add(self.view_out)
  end 
  --------------------------------------------------------------
  

  -- self.net maps a table {image_vecs, gt_seq} to word probabilities
  self.net = nn.Sequential()
  local parallel = nn.ParallelTable()
  if S==0 and not self.ATT then
    parallel:add(self.image_encoder)
  else
    parallel:add(self.image_encoder_real)-------!!!
  end
  parallel:add(self.start_token_generator)
  parallel:add(self.lookup_table)
  self.net:add(parallel)
  self.net:add(nn.JoinTable(2, 2))------!!!!!333 (1,2)->(2,2)
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
  local image_vectors = input[1]
  local gt_sequence = input[2]
  
  
  local Masks = input[3]:cuda()

  if ##image_vectors == 0 then--- if no matched pairs, don't compute loss
    self._forward_sampled = false
    return 0
  end
  
  if gt_sequence:nElement() > 0 then
    self.gt_parts = gt_sequence[{{},{16,30}}]---!!!!!
    gt_sequence = gt_sequence[{{},{1,15}}]----!!!!!
    
    
    local N, T = gt_sequence:size(1), gt_sequence:size(2)
    self._gt_with_start = gt_sequence.new(N, T + 1)
    self._gt_with_start[{{}, 1}]:fill(self.START_TOKEN)
    self._gt_with_start[{{}, {2, T + 1}}]:copy(gt_sequence)
    local mask = torch.eq(self._gt_with_start, 0)
    self._gt_with_start[mask] = self.NULL_TOKEN
    
    
    self.attention = torch.Tensor(N,T+1, 7,7):fill(0)----!!!444 (B,T+1,49)
    
    local att_pred = torch.Tensor(7,7):fill(1/49)
    for ii =1, N do---!!!!!5555
      local att_subj = torch.Tensor(7,7):zero()
      local att_obj = torch.Tensor(7,7):zero()
      
      att_subj[{{Masks[ii][1],Masks[ii][2]},{Masks[ii][3],Masks[ii][4]}}]:fill(1) ---!!subj
      att_subj = att_subj:div(att_subj:sum())----normalize & copy !!!
      att_obj[{{Masks[ii][5],Masks[ii][6]},{Masks[ii][7],Masks[ii][8]}}]:fill(1) ---!!obj
      att_obj = att_obj:div(att_obj:sum())----normalize & copy !!!
      
      --dbg()
      ---self.attention[ii][{{1,1}}] = att_subj
      --dbg()
      
      
      --self.attention[ii]:index(1, self.gt_parts[ii]:eq(1):nonzero()[1])     :copy( att_subj:expandAs( self.attention[ii]:index(1, self.gt_parts[ii]:eq(1):nonzero()[1]) )  )
              
      
      --#--code need to be optimized   !!!!!!!!!!!!!!!     
      for jj = 1, self.gt_parts:size(2) do
        if self.gt_parts[ii][jj]==1 then------------subj
          self.attention[ii][jj]:copy( att_subj)
        elseif self.gt_parts[ii][jj]==2 then--------pred
          self.attention[ii][jj]:copy(att_pred)
        elseif self.gt_parts[ii][jj]==3 then------obj
          self.attention[ii][jj]:copy(att_obj)
        end
      end
      
      ---self.attention[ii]:index(1, self.gt_parts[ii]:eq(2):nonzero()[1] ):copy(att_pred)
      ---dbg()
      ---self.attention[ii]:index(1, self.gt_parts[ii]:eq(3):nonzero()[1] ):copy(att_obj)
      
    end 

    self.attention = self.attention:view(N,T+1,7*7)  

    self.view_in:resetSize(N * (T+1), -1)--!!!333 T+2->T+1
    self.view_out:resetSize(N, T+1 , -1)--!!!333 T+2->T+1
    self.view_out2:resetSize(N, T+1 , -1)--!!!
    
    self.output = self.net:forward({{image_vectors,self.attention:cuda()},self._gt_with_start})---!!!4444
    self._forward_sampled = false
    
    return self.output

  else
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
  
  --gt_sequence = gt_sequence:index(1,self.label_idx)---!!!
  
  -- Make sure it's on CPU since we will loop over it
  local gt_sequence_long = gt_sequence:long()
  local N, T = gt_sequence:size(1), gt_sequence:size(2)
  local target = torch.LongTensor(N, T + 1):zero()--!!!333 T+2->T+1
  target[{{}, {1, T }}]:copy(gt_sequence)---!!!333 2~T+1 -> 1~~T
  for i = 1, N do
    for t = 1, T + 1 do---!!!333  t=2 -> t=1
      if target[{i, t}] == 0 then
        -- Replace the first null with an end token
        target[{i, t}] = self.END_TOKEN
        break
      end
    end
  end
  --dbg()
  return target:type(gt_sequence:type())
end

function LM:sample(image_vectors,Masks)
  local N, T = image_vectors:size(1), self.seq_length
  local seq = torch.LongTensor(N, T):zero()
  local softmax = nn.SoftMax():type(image_vectors:type())
  
  
  local attention = torch.Tensor(N, T+1, 7,7):zero()----!!!444 (B,T+1,49)
  
  local SUBJ = torch.Tensor(N,7,7)
  local PRED = torch.Tensor(N,7,7):fill(1/49)---!!!!!!
  local OBJ = torch.Tensor(N,7,7)
  
  
  
  
  if ##Masks==0 then
    attention:fill(1/49)
  else
    for ii =1, N do---!!!!!5555
      attention[{ii,{},{Masks[ii][1],Masks[ii][2]},{Masks[ii][3],Masks[ii][4]}}]
              :fill(1/(Masks[ii][2]-Masks[ii][1]+1)/(Masks[ii][4]-Masks[ii][3]+1)) ---!! start from subj
      
      SUBJ[{ii,{Masks[ii][1],Masks[ii][2]},{Masks[ii][3],Masks[ii][4]}}]
              :fill(1/(Masks[ii][2]-Masks[ii][1]+1)/(Masks[ii][4]-Masks[ii][3]+1))--!! subj
      OBJ[{ii,{Masks[ii][5],Masks[ii][6]},{Masks[ii][7],Masks[ii][8]}}]
              :fill(1/(Masks[ii][6]-Masks[ii][5]+1)/(Masks[ii][8]-Masks[ii][7]+1))--!! subj
    end
  end
  --dbg()
  attention = attention:view(N,T+1,7*7) 
  SUBJ = SUBJ:view(N,7*7)
  PRED = PRED:view(N,7*7)
  OBJ = OBJ:view(N,7*7)
  
  
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
  self.view_out2:resetSize(N, 1, -1)---!!!

  -- First timestep: image vectors, ignore output  
  if self.ATT then--!!!! 444
    image_vecs_encoded  = self.image_encoder_real:forward{image_vectors,attention:cuda()}
  else
    if ##Masks==0 then
      image_vecs_encoded  = self.image_encoder:forward(image_vectors)
    else
      image_vecs_encoded = self.image_encoder_real:forward({image_vectors,Masks})---!!!!
    end
  end
  --self.rnn:forward(image_vecs_encoded)--!!!333

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
    
    
    local imvec = image_vecs_encoded[{{(t-1)*N+1,t*N}}]---!!!333444
    
    local rnn_out = self.rnn:forward( torch.cat(imvec,wordvecs,3) )
    local scores = rnn_out[1]:view(N,-1)--!!! 444
    local att = rnn_out[2]:view(N,-1)--!!!444
    
    local idx = nil
    _, idx = torch.max(scores, 2)
    seq[{{}, t}]:copy(idx)
    
    
    probs = softmax:forward(att)
    
    
    
    local att_sum = torch.Tensor(N,7*7)
    for ii =1,N do   
      _, max_id = torch.max(probs[{ii}],1)
      --att_sum[ii] = SUBJ[ii] * probs[{ii,1}] + PRED[ii] * probs[{ii,2}] + OBJ[ii] * probs[{ii,3}]
      if max_id[1]==1 then
        att_sum[ii] = SUBJ[ii]
      elseif max_id[1]==2 then
        att_sum[ii] = PRED[ii]
      elseif max_id[1]==3 then
        att_sum[ii] = OBJ[ii]
      else
        dbg()
      end
    end
    
    if t<T+1 then
      attention[{{},t+1,{}}]:copy(att_sum)---!!!!444
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
  if ##input[3]==0 or self.ATT then ---Mask !!!!!
    net_input = {input[1], self._gt_with_start}
  else
    net_input = {{input[1]:cuda(),input[3]:cuda()}, self._gt_with_start}
  end
  
  
  if gradOutput == 0  then---if there is no box- pair detected!
    if ##input[3]==0 then
      self.gradInput = {input[1].new(0):zero()}
    else
      self.gradInput = { input[1].new(input[1]):zero() ,nil,  input[2].new(input[2]):zero() }
    end
    
  else
    
    
    if self.ATT then
      net_input = {{input[1],self.attention:cuda()}, self._gt_with_start}--!!!444
    else
      if ##input[3]==0 then ---Mask !!!!!
        net_input = {input[1], self._gt_with_start}
      else
        net_input = {{input[1]:cuda(),input[3]:cuda()}, self._gt_with_start}
      end
    end

    local tmp = self.net:backward(net_input, gradOutput, scale)

     if self.ATT then
        --self.gradInput = tmp---!!!!444
        self.gradInput = {tmp[1][1]}
     else    
      if ##input[3]==0 then--!!!!!
        self.gradInput = tmp
      else
        self.gradInput = {tmp[1][1],nil, tmp[1][2]}
      end
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

--[[
image_vectors: N x D
--]]
function LM:beamsearch(image_vectors, beam_size)
  error('Not implemented')
end
