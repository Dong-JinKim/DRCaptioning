local dbg = require("debugger")
--require 'torch'
--require 'nn'
--require 'cudnn'---!!!

--require 'densecap.DataLoader'
--require 'densecap.DenseCapModel'

--local utils = require 'densecap.utils'
local eval_utils = require 'eval.eval_utils_R_4caption'

-- Actually run evaluation
local eval_kwargs = {
  GT_captions = 'eval/VRD_model/gt_tuple_label.json',
  captions='eval/VRD_model/rlp_labels_ours.json',
  GT_boxes = 'eval/VRD_model/gt_boxes.json',
  boxes = 'eval/VRD_model/boxes.json',
  --GT_captions = 'eval/SCST/gt_tuple_label.json',
  --captions='eval/SCST/rlp_labels_ours.json',
  --GT_captions = 'data/visual-genome/gt_tuple_label_short_few.json',--long or short?
  --captions='data/visual-genome/rlp_labels_ours_few.json',
  --GT_boxes = 'data/visual-genome/gt_boxes_few.json',
  --boxes = 'data/visual-genome/boxes_few.json',
}
local eval_results = eval_utils.eval_split(eval_kwargs)
