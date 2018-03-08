local dbg = require("debugger")
require 'nn'
require 'torch-rnn'

local utils = require 'densecap.utils'


local DM, parent = torch.class('nn.DiscriminatorModel', 'nn.Module')


function DM:__init(opt)
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
  
  self.START_TOKEN = self.vocab_size + 1
  self.END_TOKEN = self.vocab_size + 1
  self.NULL_TOKEN = self.vocab_size + 2
  
  self.use_matching_loss = false

  -- self.rnn maps wordvecs of shape N x T x W to word probabilities
  -- of shape N x T x (V + 1)
  
  
  self.view_in = nn.View(1, 1, -1):setNumInputDims(3)
  self.view_out = nn.View(1, -1):setNumInputDims(2)
  
  self.nets = nn.Sequential()--------------------- for language feats
  
  self.nets:add(self.view_in)
  self.nets:add(nn.SoftMax())---- normalize with softmax before started
  self.nets:add(self.view_out)
  
  
  for i = 1, self.num_layers do
    local input_dim = self.rnn_size
    if i == 1 then
      input_dim = self.vocab_size + 1
    end
    self.nets:add(nn.LSTM(input_dim, self.rnn_size))
    if self.dropout > 0 then
      self.nets:add(nn.Dropout(self.dropout))
    end
  end
  self.select = nn.Select(2,17)
  self.nets:add(self.select)-- select the last !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  --self.nets:add(self.view_in)
  
  self.nets2 = nn.Sequential() -------------------- for image feats
  self.nets2:add(nn.Linear(8192,self.rnn_size))---!!!
  --self.nets2:add(nn.Replicate(17))
  
  
  
  local sentence = nn.Identity()()
  local im_feats = nn.Identity()()
  
  local sent_emb = self.nets(sentence)
  local im_emb   = self.nets2(im_feats)
  
  local out = nn.DotProduct()({sent_emb,im_emb})
  out = nn.Sigmoid()(out)
  
  
  self.net = nn.gModule({sentence,im_feats}, {out,sent_emb})
  
  
  self:training()
end

function DM:updateOutput(inputs)
    self.recompute_backward = true--???????????
    input = inputs[1]
    feats = inputs[2]
    -- Add a start token to the start of the gt_sequence, and replace
    -- 0 with NULL_TOKEN
    local N, T = input:size(1),input:size(2)
    --local _,label = input:max(3)
    --self.mask = torch.eq(label:squeeze(),self.END_TOKEN)
    --_,A = self.mask:max(2)
    
    -- Reset the views around the nn.Linear
    self.view_in:resetSize(N * T, -1)
    self.view_out:resetSize(N, T, -1)
    
    --dbg()
    
    self.output = self.net:forward(inputs)---!!!!!!!!!!!!!
    
    return self.output
end


function DM:updateGradInput(input, gradOutput)
  if self.recompute_backward then
    self:backward(input, gradOutput)
  end
  return self.gradInput
end


function DM:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end


function DM:backward(input, gradOutput)
  self.recompute_backward = false
  self.gradInput = self.net:backward(input, gradOutput)
  return self.gradInput
end

function DM:parameters()
  return self.net:parameters()
end


function DM:training()
  parent.training(self)
  self.net:training()
end


function DM:evaluate()
  parent.evaluate(self)
  self.net:evaluate()
end


function DM:clearState()
  self.net:clearState()
end


function DM:parameters()
  return self.net:parameters()
end

