local dbg = require("debugger")
require 'torch'
require 'nn'
require 'image'
require 'hdf5'
require 'cudnn'---!!!

require 'densecap.DataLoader'
require 'densecap.DenseCapModel'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'



local cmd = torch.CmdLine()

-- Model options
cmd:option('-checkpoint',
  'data/models/densecap/densecap-pretrained-vgg16.t7',
  'The checkpoint to evaluate')
cmd:option('-data_h5', 'data/VG-regions_R2longv3.h5', 'The HDF5 file to load data from; optional.')
cmd:option('-data_json', 'data/VG-regions-dicts_R2longv3.json', 'The JSON file to load data from; optional.')
cmd:option('-activation', 'prob_output.t7', 'The activation file to load data from; optional.')
cmd:option('-gpu', 0, 'The GPU to use; set to -1 for CPU')
cmd:option('-use_cudnn', 1, 'Whether to use cuDNN backend in GPU mode.')
cmd:option('-split', 'train', 'Which split to evaluate; either val or test.')
cmd:option('-max_images',500, 'How many images to evaluate; -1 for whole split')
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.3)
cmd:option('-num_proposals', 50)-- originally 1000
cmd:option('-boxes_per_image', 15)
NUM_OF_QUERY = 25



local function collect_query(loader,max_images)
  local query = torch.IntTensor(0)
  local count = 0
  local ix = 0
  local GT_ind = torch.IntTensor(NUM_OF_QUERY*4):zero()
  --torch.manualSeed(123)
  while true do
  --for ix = 1,25 do
    
    ix = ix + 1
    ix = torch.random(max_images)--randomly choose 1000 samples
    -- fetch the corresponding labels array
    local r0 = loader.img_to_first_box[loader.test_ix[ix]]
    local r1 = loader.img_to_last_box[loader.test_ix[ix]]
    
    local label_array = loader.labels[{ {(r0+1)/2,r1/2} }]---------------------------!!!!!! box number is double!!
    
    if #label_array:size()==2 and label_array:size(1)>3 then
      label_array = label_array[{{1,4},{1,15}}]-- take 4 caption per img
      GT_ind[{{count*4+1,(count+1)*4}}]:fill(ix)
      count = count + 1
      query = torch.cat(query,label_array,1)
    end

    if count == NUM_OF_QUERY then break end
  end 

  return query,GT_ind
end


function eval_split(kwargs)
  --dbg()
  local model = utils.getopt(kwargs, 'model')
  local loader = utils.getopt(kwargs, 'loader')
  local boxes_per_image = utils.getopt(kwargs, 'boxes_per_image')
  local split = utils.getopt(kwargs, 'split', 'test')
  local max_images = utils.getopt(kwargs, 'max_images',1000)
  local dtype = utils.getopt(kwargs, 'dtype', 'torch.FloatTensor')
  local split_to_int = {train=0,val=1, test=2}
  split = split_to_int[split]
  print('using split ', split)

  
  local err_map = torch.FloatTensor(NUM_OF_QUERY*4,max_images):zero()
  local max_box = torch.LongTensor(NUM_OF_QUERY*4,max_images):zero()--- box-pair index that has maximum score
  
  model:evaluate()
  loader:resetIterator(split)
  
  local TRANSF = false---!!!777
   
  local counter = 0
  local softmax = nn.SoftMax()
  
  local query,GT_ind = collect_query(loader,max_images)
  local target = model.nets.language_model:getTarget(query):cuda()--!!!!
  --dbg()
  for iid = 1,max_images do---1000
    counter = counter + 1
    
    local ix = loader.test_ix[iid]
    -- Grab a batch of data and convert it to the right dtype
    local data = {}
    local opt = {split=split, iterate=true}
    --local img, gt_boxes, gt_labels, info, _ = loader:getBatch(loader_kwargs)
    
    
    local  img = loader.h5_file:read('/images'):partial({ix,ix},{1,loader.num_channels},
                            {1,loader.max_image_size},{1,loader.max_image_size})
    -- crop image to its original width/height, get rid of padding, and dummy first dim
    img = img[{ 1, {}, {1,loader.image_heights[ix]}, {1,loader.image_widths[ix]} }]
    img = img:float() -- convert to float
    img = img:view(1, img:size(1), img:size(2), img:size(3)) -- batch the image
    img:add(-1, loader.vgg_mean:expandAs(img)) -- subtract vgg mean
    -- fetch the corresponding labels array
    local r0 = loader.img_to_first_box[ix]
    local r1 = loader.img_to_last_box[ix]
    
    local gt_labels = loader.labels[{ {(r0+1)/2,r1/2} }]---------------------------!!!!!! box number is double!!
    local gt_boxes = loader.boxes[{ {r0,r1} }]
    -- batch the boxes and labels
    gt_labels = gt_labels:view(1, gt_labels:size(1), gt_labels:size(2))
    gt_boxes = gt_boxes:view(1, gt_boxes:size(1), gt_boxes:size(2))
    local filename = loader.info.idx_to_filename[tostring(ix)] -- json is loaded with string keys
    local info = { {filename = filename, 
                        split_bounds = {ri, #loader.test_ix},
                        width = w, height = h, ori_width = ow, ori_height = oh} }
    --dbg()
    local data = {
      image = img:type(dtype),
      gt_boxes = gt_boxes:type(dtype),
      gt_labels = gt_labels:type(dtype),
    }
    info = info[1] -- Since we are only using a single image
    

    
    -- Call forward_backward to compute losses
    model.timing = false
    model.dump_vars = false
    model.cnn_backward = false
    
    model:training()
    -- Run the model forward
    model:setGroundTruth(data.gt_boxes, data.gt_labels)
    local out = model:forward(data.image)
    local lm_output = out[5]
    
    --if iid==19 then
    --dbg() 
    --end
    --dbg()
    
    if #lm_output==2 then
      lm_output = lm_output[1]----remove POS classifier output
    end
    --dbg()
    
    -- lm_output B*17*V
    -- target 100,17
    --dbg()
    local B, T, V = lm_output:size(1), lm_output:size(2), lm_output:size(3)
    
    target[target:eq(0)]=V--fill <END> for the rest of the sentence
    
    lm_output = lm_output:view(B*T,-1)
    lm_output = softmax:forward(lm_output:double())--applyting softmax
    lm_output = lm_output:view(B,T,-1):cuda()
    
    
    
    
    
    lm_output[{{},{},V}]:fill(1)-- we dont condier <END> token
    --dbg()
    for tid = 1,NUM_OF_QUERY*4 do---query
      
      local lm_output_tmp = lm_output.new(B,T):fill(1)---pooling first !!! 
      --local lm_output_tmp = lm_output.new(T):fill(1)--  max first !!!
      --dbg()
      for wid = 1,T do--word
        --lm_output_tmp[{{},wid}]:copy(lm_output[{{},{},target[{tid,wid}]}]:max(2))--existence-pooling first!!
        lm_output_tmp[{{},wid}]:copy(lm_output[{{},wid,target[{tid,wid}]}])---cross entropy and (pooling) first!!!
        --lm_output_tmp[wid] = lm_output[{{},wid,target[{tid,wid}]}]:max() ---- max first!
      end
      
      lm_output_tmp = lm_output_tmp:prod(2):view(B)--computing prob for all caption---- (max next)(prod or sum?)
      local prob,max_ind  = lm_output_tmp:max(1)------------------------------------------------ (max next)
      --local prob = lm_output_tmp:prod(1)[1] -- (pooling next)(prod or sum?)
      err_map[tid][iid] = prob[1]
      max_box[tid][iid] = max_ind[1]
      
    end



    -- Print a message to the console
    local msg = 'Processed image %s (%d / %d) of split %d'
    local num_images = info.split_bounds[2]
    print(string.format(msg, info.filename, counter, max_images, split, num_boxes))

    
  end
  local sorted,ind = (-err_map):sort(2)--top k rank, 3200-dim between 1~100
  --GT_ind
  
  local ranks = GT_ind.new(NUM_OF_QUERY*4)
  --local rank_sum = GT_ind.new(NUM_OF_QUERY):zero()
  --dbg()
  for qid = 1,NUM_OF_QUERY*4 do
    local tmp = ind[qid]:eq(GT_ind[qid]):nonzero()
      ranks[qid] = tmp[1][1]
      --rank_sum[torch.ceil(qid/4)] = rank_sum[torch.ceil(qid/4)] + ranks[qid]

  end
  --dbg()
  --local _,good_indexes = rank_sum:sort()-- selected image(for query) index based on performance
  --local selected_index = torch.cat({good_indexes[{{1,25}}]*4-3 , good_indexes[{{1,25}}]*4-2, good_indexes[{{1,25}}]*4-1, good_indexes[{{1,25}}]*4},1) --selected query index
  --dbg()
  --ranks = ranks:index(1,selected_index )
  
  if false then
    local good1 = ranks:lt(5+1)-- number between 1~3200  --ranks:eq(5):nonzero()--
    
    local good2 = (15-query:eq(0):sum(2)):gt(4)--- GT length >= 5
    
    good1 = good1:cmul(good2)
    good1 = good1:nonzero()
    
    local utils = require 'densecap.utils'
    
    local JSON = utils.read_json(loader.json_file)
    
    
    local query_sentences = model.nets.language_model:decodeSequence(query)
    fd = io.open('retrieved_regions_rank5_longv3_70proposal_model1.txt','w') -- retrieved_regions_rank3_long_60proposal_model2.txt','w'
    for qid = 1,good1:nElement() do
      local GT_ix = JSON.idx_to_filename[string.format('%d',loader.test_ix[GT_ind[good1[qid][1]]])]
      local ix = torch.LongTensor(6)
      local pid = torch.LongTensor(6)
      for rid = 1,6 do
        --dbg()
        ix[rid] = loader.test_ix[ind[good1[qid][1]][rid]]
        pid[rid] = max_box[good1[qid][1]][ind[good1[qid][1]][rid]]
        --dbg()
     end
      --dbg()
      print(string.format('Query :%s %s :  top1: %s,%d / top2: %s,%d  / top3: %s,%d  / top4: %s,%d  / top5: %s,%d / top6: %s,%d \n',
        GT_ix, query_sentences[good1[qid][1]], JSON.idx_to_filename[string.format('%d',ix[1])],pid[1] , 
                                     JSON.idx_to_filename[string.format('%d',ix[2])],pid[2] ,
                                     JSON.idx_to_filename[string.format('%d',ix[3])],pid[3] ,
                                     JSON.idx_to_filename[string.format('%d',ix[4])],pid[4] , 
                                     JSON.idx_to_filename[string.format('%d',ix[5])],pid[5] ,
                                     JSON.idx_to_filename[string.format('%d',ix[6])],pid[6] ))
    
      
      fd:write(string.format('Query :%s %s :  top1: %s,%d / top2: %s,%d  / top3: %s,%d  / top4: %s,%d  / top5: %s,%d / top6: %s,%d \n',
        GT_ix, query_sentences[good1[qid][1]], JSON.idx_to_filename[string.format('%d',ix[1])],pid[1] , 
                                     JSON.idx_to_filename[string.format('%d',ix[2])],pid[2] ,
                                     JSON.idx_to_filename[string.format('%d',ix[3])],pid[3] ,
                                     JSON.idx_to_filename[string.format('%d',ix[4])],pid[4] , 
                                     JSON.idx_to_filename[string.format('%d',ix[5])],pid[5] ,
                                     JSON.idx_to_filename[string.format('%d',ix[6])],pid[6] ))
      
    
    end
    fd:close()
    --dbg()
  end
  
  
  
  
  --local json_out = {}
  --json_out.captions = evaluator.captions
  --json_out.opt = model.opt
  --utils.write_json('relcap_statistics_tmp.json', json_out)----!!!!333
  
  
  
  
  
  
  
  
  
  local r1 = 100.0* ( ranks:lt(1+1):sum() )/ ranks:size(1)
  local r5 = 100.0* ( ranks:lt(5+1):sum() )/ ranks:size(1)
  local r10 = 100.0* ( ranks:lt(10+1):sum() )/ ranks:size(1)
  local medr = ranks:median()[1]
  local meanr = ranks:sum()/ranks:size(1)

  
  print(string.format('Text to image: R1:%.1f, R5:%.1f, R10:%.1f, med:%.1f, mean:%.1f', r1, r5, r10, medr, meanr))
  
  
  
  
  
  
  return {r1, r5, r10, medr, meanr}
end


local function main()
  local opt = cmd:parse(arg)
  
  local loader = DataLoader(opt)
  
  
  -- Load and set up the model
  local dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn)
  local checkpoint = torch.load(opt.checkpoint)
  local model = checkpoint.model
  print 'Loaded model'
  model:convert(dtype, use_cudnn)
  
  model.nets.localization_layer.nets.box_sampler_helper.box_sampler.batch_size =opt.num_proposals
  
  model:setTestArgs{
    rpn_nms_thresh = opt.rpn_nms_thresh,
    final_nms_thresh = opt.final_nms_thresh,
    num_proposals = opt.num_proposals,
  }
  model:evaluate()
  
  -- Actually run evaluation
  local eval_kwargs = {
    model=model,
    loader=loader,
    split=opt.split,
    dtype=dtype,
    boxes_per_image = opt.boxes_per_image,
    max_images = opt.max_images,
  }
  
  for ii = 1,1000 do
  result = eval_split(eval_kwargs)
  r1, r5, r10, medr, meanr = result[1] , result[2] , result[3] , result[4] , result[5]
  if r1>27  then
    fd = io.open(string.format('retrieval_result/%d-%d.txt',opt.gpu,ii),'w')
    fd:write(string.format(' R1  /  R5  /  R10 /  med /  mean  \n%.1f / %.1f / %.1f / %.1f  / %.1f',r1,r5,r10,medr,meanr))
    fd:close()
  end
  collectgarbage()
  end
  
end

main()
