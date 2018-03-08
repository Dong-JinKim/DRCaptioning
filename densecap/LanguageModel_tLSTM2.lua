local dbg = require("debugger")
require 'nn'
require 'torch-rnn'
require 'nngraph'

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
  local ATT = false
  local ATT_temp = false
  
  -- For mapping from image vectors to word vectors
  self.image_encoder = nn.Sequential()  
  
  --self.image_encoder:add(nn.SpatialAveragePooling(7,7))------!!!!2222(B, 512,7,7)->(B, 512,1,1)
  --self.image_encoder:add(nn.View(-1):setNumInputDims(3))--!!!222 (B, 512,1,1) -> (B,512)
  
  self.image_encoder:add(nn.View(-1):setNumInputDims(3))-- (B, 512,7,7)->(B,25088)
  self.image_encoder:add(nn.Linear(25088,512))
  self.image_encoder:add(nn.ReLU(true))
  --self.image_encoder:add(nn.Dropout(0.5))
  self.image_encoder:add(nn.Linear(512,512))
  self.image_encoder:add(nn.ReLU(true))
  --self.image_encoder:add(nn.Dropout(0.5))


  if S==0 then
    self.image_encoder:add(nn.Linear(D, W))----!!!222 (B,4096)-> (B,512)
    self.image_encoder:add(nn.ReLU(true))
    self.image_encoder:add(nn.View(1, -1):setNumInputDims(1)) 
  elseif S==6 then
    self.image_encoder_real = nn.Sequential()
    
    self.spatial_encoder = nn.Sequential()
    self.spatial_encoder:add(nn.Linear(6,64))
    self.spatial_encoder:add(nn.ReLU(true))
    
    local image_encoder12 = nn.ParallelTable()---merge imvec and mask activation
    image_encoder12:add(self.image_encoder)
    image_encoder12:add(self.spatial_encoder)
    
    
    self.image_encoder_real:add(image_encoder12)
    self.image_encoder_real:add(nn.JoinTable(2,2))
    
    self.image_encoder_real:add(nn.Linear(512+64,W))--!!!666666
    self.image_encoder_real:add(nn.ReLU(true))
    self.image_encoder_real:add(nn.View(1, -1):setNumInputDims(1)) 
  else
    dbg()
  end
  
  -----------------------2-----------------
  self.image_encoder2 = nn.Sequential()  
  
  --self.image_encoder2:add(nn.SpatialAveragePooling(7,7))------!!!!2222(B, 512,7,7)->(B, 512,1,1)
  --self.image_encoder2:add(nn.View(-1):setNumInputDims(3))--!!!222 (B, 512,1,1) -> (B,512)
  
  self.image_encoder2:add(nn.View(-1):setNumInputDims(3))-- (B, 512,7,7)->(B,25088)
  self.image_encoder2:add(nn.Linear(25088,512))
  self.image_encoder2:add(nn.ReLU(true))
  
  --self.image_encoder2:add(nn.SpatialConvolution(512, 16, 1, 1))
  --self.image_encoder2:add(nn.ReLU(true))
  --self.image_encoder2:add(nn.View(-1):setNumInputDims(3))-- (B, 16,7,7)->(B,784)
  
  
  self.image_encoder2:add(nn.Linear(512, W))----!!!222 (B,4096)-> (B,512)
  self.image_encoder2:add(nn.ReLU(true))
  self.image_encoder2:add(nn.View(1, -1):setNumInputDims(1))
  -----------------------------------------
  self.image_encoder3 = nn.Sequential()  
  
  --self.image_encoder3:add(nn.SpatialAveragePooling(7,7))------!!!!2222(B, 512,7,7)->(B, 512,1,1)
  --self.image_encoder3:add(nn.View(-1):setNumInputDims(3))--!!!222 (B, 512,1,1) -> (B,512)
  
  self.image_encoder3:add(nn.View(-1):setNumInputDims(3))-- (B, 512,7,7)->(B,25088)
  self.image_encoder3:add(nn.Linear(25088,512))
  self.image_encoder3:add(nn.ReLU(true))
  
  --self.image_encoder3:add(nn.SpatialConvolution(512, 16, 1, 1))
  --self.image_encoder3:add(nn.ReLU(true))
  --self.image_encoder3:add(nn.View(-1):setNumInputDims(3))-- (B, 16,7,7)->(B,784)
  
  self.image_encoder3:add(nn.Linear(512, W))----!!!222 (B,4096)-> (B,512)
  self.image_encoder3:add(nn.ReLU(true))
  self.image_encoder3:add(nn.View(1, -1):setNumInputDims(1))

  
  self.START_TOKEN = self.vocab_size + 1
  self.END_TOKEN = self.vocab_size + 1
  self.NULL_TOKEN = self.vocab_size + 2

  -- For mapping word indices to word vectors
  local V, W = self.vocab_size, self.input_encoding_size
  self.lookup_table = nn.LookupTable(V + 2, W)----!!! 
  self.lookup_table2 = nn.LookupTable(V + 2, W)----!!! 
  self.lookup_table3 = nn.LookupTable(V + 2, W)----!!! 
  
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
  if S==0 then
    parallel1:add(self.image_encoder)
  elseif S==6 then
    parallel1:add(self.image_encoder_real)
  else
    dbg()
  end
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
    self.rnn2:add(nn.LSTM(input_dim, self.rnn_size))
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
  self.rnn3 = nn.Sequential()---------LSTM3
  for i = 1, self.num_layers do
    local input_dim = self.rnn_size
    if i == 1 then
      input_dim = self.input_encoding_size
    end
    self.rnn3:add(nn.LSTM(input_dim, self.rnn_size))
    if self.dropout > 0 then
      self.rnn3:add(nn.Dropout(self.dropout))
    end
  end
  local parallel3 = nn.ParallelTable()
  parallel3:add(self.image_encoder3)
  parallel3:add(self.start_token_generator)
  parallel3:add(self.lookup_table3)
  self.input3 = nn.Sequential()
  self.input3:add(parallel3)
  self.input3:add(nn.JoinTable(1, 2))
  self.input3:add(self.rnn3)
  ------------------------------------------------------------------------------
  self.view_in = nn.View(1, 1, -1):setNumInputDims(3)
  self.view_out = nn.View(1, -1):setNumInputDims(2)

  
  local combine = nn.ParallelTable()---merge imvec and mask activation
  combine:add(self.input1)
  combine:add(self.input2)
  combine:add(self.input3)
  
  -- self.net maps a table {image_vecs, gt_seq} to word probabilities
  self.net = nn.Sequential()
  self.net:add(combine)
  
  --self.net:add(nn.JoinTable(2, 2))----( concat/attention)
  self.net:add(nn.CAddTable())---(sum),
  --self.net:add(nn.CMulTable())---(mul)


  

  if not(ATT) then
    --(1)-------------------------------------------------------- ------------------  not attention
    self.out=nn.Sequential()
    
    if not (ATT_temp) then
    
      self.out:add(self.view_in)-- (B,1,3*H)->(B,3*H)
      self.out:add(nn.Linear(H, H))
      self.out:add(nn.ReLU(true))
      self.out:add(nn.Dropout(0.5))---!!!
    
    else
      self.out:add(nn.View(-1,3,H):setNumInputDims(3))-- (B,T,3*H)->(BT,3,H)
      --(2)----------------------------------------------------------------------------- attention
      self.att = nn.Sequential()--branch for attention
      self.att:add(nn.View(-1,H):setNumInputDims(3))-- (BT,3,H)->(BT3,H)
      self.att:add(nn.Linear(H,1))-- (3BT,H)->(3BT,1)
      self.att:add(nn.View(-1,3):setNumInputDims(2))-- (3BT,1)->(BT,3)
      self.att:add(nn.SoftMax())
      self.att:add(nn.View(1,-1):setNumInputDims(1))-- (BT,3)->(BT,1,3)
      
      local feat = nn.Sequential()--branch for feature
      feat:add(nn.Identity())--(BT,3,H)
      
      local ATTention = nn.ConcatTable()
      ATTention:add(self.att)
      ATTention:add(feat)
      self.out:add(ATTention)
      self.out:add(nn.MM())-- (BT,1,3)*(BT,3,H) -> (BT,1,H)
      self.out:add(nn.View(-1):setNumInputDims(2))-- (BT,1,H)->(BT,H)
      
    end
    
     ----------------------------------------------------------
    self.view_out2 = nn.View(1, -1):setNumInputDims(2)--!!!
    local cap_branch = nn.Sequential()--branch for word output
    cap_branch:add(nn.Linear(H, V + 1))
    cap_branch:add(self.view_out)
    
    local cls_branch = nn.Sequential()-- branch for attention
    cls_branch:add(nn.Linear(H, 3))-- parts class !!!!!
    cls_branch:add(self.view_out2)
    
    local branch = nn.ConcatTable()
    branch:add(cap_branch)
    branch:add(cls_branch)
    
    self.out:add(branch)
    --------------------------------------------------------
  
  else
    --(2)----------------------------------------------------------------------------- attention
    local out_input = nn.View(-1,3,H):setNumInputDims(3)()-- (B,T,3*H)->(BT,3,H)
    
    self.att = nn.Sequential()--branch for attention
    self.att:add(nn.View(-1,H):setNumInputDims(3))-- (BT,3,H)->(BT3,H)
    self.att:add(nn.Linear(H,1))-- (3BT,H)->(3BT,1)
    self.att:add(nn.View(-1,3):setNumInputDims(2))-- (3BT,1)->(BT,3)

    local attention_legit = self.att(out_input)--(BT,3)
    
    self.att2 = nn.Sequential()--branch for attention
    self.att2:add(nn.SoftMax())
    self.att2:add(nn.View(1,-1):setNumInputDims(1))-- (BT,3)->(BT,1,3)
    
    local feat = nn.Sequential()--branch for feature
    feat:add(nn.Identity())--(BT,3,H)

    
    local ATTention = nn.ParallelTable()
    ATTention:add(self.att2)
    ATTention:add(feat)
    
    local output_layer = nn.Sequential()
    output_layer:add(ATTention)
    output_layer:add(nn.MM())-- (BT,1,3)*(BT,3,H) -> (BT,1,H)
    output_layer:add(nn.View(-1):setNumInputDims(2))-- (BT,1,H)->(BT,H)
    output_layer:add(nn.Linear(H, V + 1)) -- (BT, H) -> (BT, V)
    output_layer:add(self.view_out) --
    
    self.view_out2 = nn.View(1, -1):setNumInputDims(2)--!!!
    
    local cap_output = output_layer{attention_legit,out_input}
    local cls_output = self.view_out2(attention_legit)
    
    local inputs = {out_input}
    local outputs = {cap_output,cls_output}
    --local outputs = {cap_output}--,cls_output}
    --dbg()
    self.out = nn.gModule(inputs, outputs)
  end
  
  
 

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
    self.gt_parts = gt_sequence[{{},{16,30}}]---!!!!!666
    gt_sequence = gt_sequence[{{},{1,15}}]----!!!!!666
    
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
    self.view_out2:resetSize(N, T+2 , -1)--!!!666
    
    if ##Masks==0 then
      self.output = self.net:updateOutput{{union_vectors, self._gt_with_start},{subj_vectors, self._gt_with_start},{obj_vectors, self._gt_with_start}}
    else
      --dbg()
      self.output = self.net:updateOutput{{{union_vectors,Masks}, self._gt_with_start},{subj_vectors, self._gt_with_start},{obj_vectors, self._gt_with_start}}
    end
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
  local POS = torch.LongTensor(N,T):zero()---POS class
  local transfer = torch.FloatTensor(N,T,3):zero()--------!!!!!!!!777 weight transfer param for tLSTM
  
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
  for i = 1, #self.rnn3 do---!!!333
    local layer = self.rnn3:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = true
    end
  end

  -- Reset view sizes
  self.view_in:resetSize(N, -1)
  self.view_out:resetSize(N, 1, -1)
  self.view_out2:resetSize(N, 1, -1)---!!!666

  -- First timestep: image vectors, ignore output
  
  local union_vecs_encoded
  local subj_vecs_encoded
  local obj_vecs_encoded
  if ##Masks==0 then
    union_vecs_encoded  = self.image_encoder:forward(union_vectors)
  else
    --dbg()
    union_vecs_encoded  = self.image_encoder_real:forward{union_vectors,Masks}
  end
  subj_vecs_encoded  = self.image_encoder2:forward(subj_vectors)
  obj_vecs_encoded  = self.image_encoder3:forward(obj_vectors)
  
  self.rnn1:forward(union_vecs_encoded)---!!!333
  self.rnn2:forward(subj_vecs_encoded)---!!!333
  self.rnn3:forward(obj_vecs_encoded)---!!!333
  
  local GO_STOP=torch.LongTensor(N):fill(1)-- based on <END> token, weather add transfer activation or not. 

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
    local wordvecs3 = self.lookup_table3:forward(words)
    
    
    local rnnout1 = self.rnn1:forward(wordvecs)
    local rnnout2 = self.rnn2:forward(wordvecs2)
    local rnnout3 = self.rnn3:forward(wordvecs3)
    
    
    ---concat/attention
    --local scores_tmp = self.out:forward(  torch.cat( {rnnout1, rnnout2,rnnout3} ,3)    )
    --sum
    local scores_tmp = self.out:forward(  rnnout1 + rnnout2 + rnnout3     )--!!!
    --mul
    --local scores_tmp = self.out:forward(  torch.cmul(torch.cmul(rnnout1,rnnout2),srnnout3 )    )--!!!
    
    
    local scores = scores_tmp[1]:view(N, -1)
    local POS_scores = scores_tmp[2]:view(N, -1)
    
    ---------------------------------------------------------------------------------------------------------------------------------------
    --local norms = torch.cat({torch.abs(rnnout1):max(3):float(),torch.abs(rnnout2):max(3):float(),torch.abs(rnnout3):max(3):float()},3)--output/max
    --local norms = torch.cat({rnnout1:norm(2,3),rnnout2:norm(2,3),rnnout3:norm(2,3)},3)---!!!777 -output/L2
    --------------------------------------------------------------------------------------------------------------------------------------
    --local norms = torch.cat({torch.abs(self.rnn1:get(1).h0):max(2):float(),torch.abs(self.rnn2:get(1).h0):max(2):float(),torch.abs(self.rnn3:get(1).h0):max(2):float()},2)---hidden/max
    local norms = torch.cat({self.rnn1:get(1).h0:norm(2,2),self.rnn2:get(1).h0:norm(2,2),self.rnn3:get(1).h0:norm(2,2)},2)---!!!777--hidden/L2
    --------------------------------------------------------------------------------------------------------------------------------------
    
    local idx = nil
    if self.sample_argmax then
      _, idx = torch.max(scores, 2)
    else
      local probs = softmax:forward(scores)
      idx = torch.multinomial(probs, 1):view(-1):long()
    end
    
    
    
    if transfer then
      for ii=1,N do
        if idx[ii][1] == self.END_TOKEN then--if met <END> then don't add it anny more
          GO_STOP[ii] = 0
        end
        --transfer[ii][t]:copy( torch.FloatTensor{{  norms[ii][1][2]*GO_STOP[ii] , norms[ii][1][1]*GO_STOP[ii] ,norms[ii][1][3]*GO_STOP[ii] }}  )--output
        transfer[ii][t]:copy( torch.FloatTensor{{  norms[ii][2]*GO_STOP[ii] , norms[ii][1]*GO_STOP[ii] ,norms[ii][3]*GO_STOP[ii] }}  )--hidden
      end
    end
    
    
    _,max_POS = POS_scores:max(2)
    POS[{{}, t}]:copy(max_POS )
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
  -- After sampling stop remembering states
  for i = 1, #self.rnn3 do
    local layer = self.rnn3:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = false
    end
  end
  
  
  
  --dbg()
  self.output = seq
  --self.output = {seq,transfer,POS}---!!!777
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
    --if ##input[3]==0 then
      self.gradInput = {input[1]:new():zero(),nil,input[3]:new():zero(), input[4]:new():zero()}
    --else
    --  self.gradInput = { input[1]:new():zero() ,nil,  input[2]:new():zero() ,{input[4][1]:new():zero(),input[4][2]:new():zero()}}
    --end
  else
    if ##input[3]==0 then ---Mask !!!!!
      net_input = { {input[1]:cuda(), self._gt_with_start},{input[4][1]:cuda(), self._gt_with_start},{input[4][2]:cuda(), self._gt_with_start} }
    else
      net_input = { {{input[1]:cuda(),input[3]}, self._gt_with_start},{input[4][1]:cuda(), self._gt_with_start},{input[4][2]:cuda(), self._gt_with_start} }
    end
  
     local tmp = self.net:backward(net_input, gradOutput, scale)
     if ##input[3]==0 then--!!!!!
       self.gradInput = {tmp[1][1],nil,nil,{tmp[2][1],tmp[3][1]} }
     else
       self.gradInput = {tmp[1][1][1],nil , tmp[1][1][2],{tmp[2][1],tmp[3][1]}}---!!!777
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

