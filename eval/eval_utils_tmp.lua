local dbg = require("debugger")
local cjson = require 'cjson'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'

local eval_utils = {}


--[[
Evaluate a DenseCapModel on a split of data from a DataLoader.

Input: An object with the following keys:
- model: A DenseCapModel object to evaluate; required.
- loader: A DataLoader object; required.
- split: Either 'val' or 'test'; default is 'val'
- max_images: Integer giving the number of images to use, or -1 to use the
  entire split. Default is -1.
- id: ID for cross-validation; default is ''.
- dtype: torch datatype to which data should be cast before passing to the
  model. Default is 'torch.FloatTensor'.
--]]
function eval_utils.eval_split(kwargs)
  local model = utils.getopt(kwargs, 'model')
  local loader = utils.getopt(kwargs, 'loader')
  local split = utils.getopt(kwargs, 'split', 'val')
  local max_images = utils.getopt(kwargs, 'max_images', -1)
  local id = utils.getopt(kwargs, 'id', '')
  local dtype = utils.getopt(kwargs, 'dtype', 'torch.FloatTensor')
  assert(split == 'val' or split == 'test', 'split must be "val" or "test"')
  local split_to_int = {val=1, test=2}
  split = split_to_int[split]
  print('using split ', split)
  
  model:evaluate()
  loader:resetIterator(split)
  local evaluator = DenseCaptioningEvaluator{id=id}
  

  local counter = 0
  local all_losses = {}
  local num_box = 0
  local num_caption = 0
  
  while true do
    counter = counter + 1
    
    -- Grab a batch of data and convert it to the right dtype
    local data = {}
    local loader_kwargs = {split=split, iterate=true}
    local img, gt_boxes, gt_labels, info, _ = loader:getBatch(loader_kwargs)
    local data = {
      image = img:type(dtype),
      gt_boxes = gt_boxes:type(dtype),
      gt_labels = gt_labels:type(dtype),
    }
    info = info[1] -- Since we are only using a single image

    -- Call forward_test to make predictions, and pass them to evaluator
    local boxes, logprobs, captions = model:forward_test(data.image)
    num_box = num_box + boxes:size(1)
    num_caption = num_caption + #captions
    
    local gt_captions = model.nets.language_model:decodeSequence(gt_labels[1])
    evaluator:addResult(logprobs, boxes, captions, gt_boxes[1], gt_captions)
    
    -- Print a message to the console
    local msg = 'Processed image %s (%d / %d) of split %d, detected %d regions'
    local num_images = info.split_bounds[2]
    if max_images > 0 then num_images = math.min(num_images, max_images) end
    local num_boxes = boxes:size(1)
    print(string.format(msg, info.filename, counter, num_images, split, num_boxes))

    -- Break out if we have processed enough images
    if max_images > 0 and counter >= max_images then break end
    if info.split_bounds[1] == info.split_bounds[2] then break end
  end
 
  
  
  
  local utils = require 'densecap.utils'
  local json_out = {}
  json_out.captions = evaluator.captions
  json_out.opt = model.opt
  utils.write_json('relcap_statistics_proposed75.json', json_out)----!!!!333

  
  local out = {
    loss_results=loss_results,
    ap_results=ap_results,
  }
  return out
end


function eval_utils.score_captions(records)
  -- serialize records to json file
  utils.write_json('eval/input.json', records)
  -- invoke python process 
  os.execute('python eval/meteor_bridge.py')
  -- read out results
  local blob = utils.read_json('eval/output.json')
  return blob
end


local function pluck_boxes(ix, boxes, text)
  -- ix is a list (length N) of LongTensors giving indices to boxes/text. Use them to do merge
  -- this is done because multiple ground truth annotations can be on top of each other, and
  -- we want to instead group many overlapping boxes into one, with multiple caption references.
  -- return boxes Nx4, and text[] of length N

  local N = #ix
  local new_boxes = torch.zeros(N, 4)
  local new_text = {}

  for i=1,N do
    
    local ixi = ix[i]
    local n = ixi:nElement()
    local bsub = boxes:index(1, ixi)
    local newbox = torch.mean(bsub, 1)
    new_boxes[i] = newbox

    local texts = {}
    if text then
      for j=1,n do
        table.insert(texts, text[ixi[j]])
      end
    end
    table.insert(new_text, texts)
  end

  return new_boxes, new_text
end


local DenseCaptioningEvaluator = torch.class('DenseCaptioningEvaluator')
function DenseCaptioningEvaluator:__init(opt)
  self.all_logprobs = {}
  self.records = {}
  self.n = 1
  self.npos = 0
  self.id = utils.getopt(opt, 'id', '')
  self.captions = {}---!!!333
end

function DenseCaptioningEvaluator:addResult(logprobs, boxes, text, target_boxes, target_text)
  table.insert(self.captions,text)
end


function DenseCaptioningEvaluator:numAdded()
  return self.n - 1
end

return eval_utils
