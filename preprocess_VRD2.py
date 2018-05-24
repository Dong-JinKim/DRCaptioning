# coding=utf8

import argparse, os, json, string
from collections import Counter
from Queue import Queue
from threading import Thread, Lock

from math import floor
import h5py
import numpy as np
from scipy.misc import imread, imresize

import pdb

"""
This file expects a JSON file containing ground-truth regions and captions
in the same format as the region descriptions file from the Visual Genome
website. Concretely, this is a single large JSON file containing a list;
each element of the list describes a single image and has the following
format:

{
  "id": [int], Unique identifier for this image,
  "regions": [
    {
      "id": [int] Unique identifier for this region,
      "image": [int] ID of the image to which this region belongs,
      "height": [int] Height of the region in pixels,
      "width": [int] Width of the region in pixels,
      "phrase": [string] Caption for this region,
      "x": [int] x-coordinate of the upper-left corner of the region,
      "y": [int] y-coordinate of the upper-left corner of the region,
    },
    ...
  ]
}

We assume that all images are on disk in a single folder, and that
the filename for each image is the same as its id with a .jpg extension.

This file will be preprocessed into an HDF5 file and a JSON file with
some auxiliary information. The captions will be tokenized with some
basic preprocessing (split by words, remove special characters).

Note, in general any indices anywhere in input/output of this file are 1-indexed.

The output JSON file is an object with the following elements:
- token_to_idx: Dictionary mapping strings to integers for encoding tokens, 
                in 1-indexed format.
- filename_to_idx: Dictionary mapping string filenames to indices.
- idx_to_token: Inverse of the above.
- idx_to_filename: Inverse of the above.

The output HDF5 file has the following format to describe N images with
M total regions:

- images: uint8 array of shape (N, 3, image_size, image_size) of pixel data,
  in BDHW format. Images will be resized so their longest edge is image_size
  pixels long, aligned to the upper left corner, and padded with zeros.
  The actual size of each image is stored in the image_heights and image_widths
  fields.
- image_heights: int32 array of shape (N,) giving the height of each image.
- image_widths: int32 array of shape (N,) giving the width of each image.
- original_heights: int32 array of shape (N,) giving the original height of
  each image.
- original_widths: int32 array of shape (N,) giving the original width of
  each image.
- boxes: int32 array of shape (M, 4) giving the coordinates of each bounding box.
  Each row is (xc, yc, w, h) where yc and xc are center coordinates of the box,
  and are one-indexed.
- lengths: int32 array of shape (M,) giving lengths of label sequence for each box
- captions: int32 array of shape (M, L) giving the captions for each region.
  Captions in the input with more than L = --max_token_length tokens are
  discarded. To recover a token from an integer in this matrix,
  use idx_to_token from the JSON output file. Padded with zeros.
- img_to_first_box: int32 array of shape (N,). If img_to_first_box[i] = j then
  captions[j] and boxes[j] give the first annotation for image i
  (using one-indexing).
- img_to_last_box: int32 array of shape (N,). If img_to_last_box[i] = j then
  captions[j] and boxes[j] give the last annotation for image i
  (using one-indexing).
- box_to_img: int32 array of shape (M,). If box_to_img[i] = j then then
  regions[i] and captions[i] refer to images[j] (using one-indexing).
"""
def build_vocab(data, min_token_instances, verbose=True):
  """ Builds a set that contains the vocab. Filters infrequent tokens. """
  token_counter = Counter()
  object_counter = Counter()#----!!!
  predicate_counter = Counter()#---!!!!
  
  for img in data:
    for region in img['relationships']:
      if region['tokens'] is not None:
        token_counter.update(region['tokens'])
      #pdb.set_trace()
      #if region['predicate'] is not None:
      #  predicate_counter.update([region['predicate']])
      #if region['subject'] is not None:
      #  object_counter.update([region['subject']['name']])#---!!!
      #if region['object'] is not None:
      #  object_counter.update([region['object']['name']])#----!!!!
      print("%s - %s - %s"%(region['subject']['name'],region['predicate'],region['object']['name']))
      
    
  vocab = set()
  vocab_pred = set()#---!!!
  vocab_obj = set()#---!!!!
  for token, count in token_counter.iteritems():
    if count >= min_token_instances:
      vocab.add(token)
  #for token, count in predicate_counter.iteritems():#---!!!
  #  vocab_pred.add(token)
  #for token, count in object_counter.iteritems():#---!!!
  #  vocab_obj.add(token)        
  
  if verbose:
    print ('Keeping %d / %d tokens with enough instances'
              % (len(vocab), len(token_counter)))
  
  if len(vocab) < len(token_counter):
    vocab.add('<UNK>')
    if verbose:
      print('adding special <UNK> token.')
  else:
    if verbose: 
      print('no <UNK> token needed.')

  return vocab,vocab_pred,vocab_obj


def build_vocab_dict(vocab):
  token_to_idx, idx_to_token = {}, {}
  next_idx = 1

  for token in vocab:
    token_to_idx[token] = next_idx
    idx_to_token[next_idx] = token
    next_idx = next_idx + 1
    
  return token_to_idx, idx_to_token


def encode_caption(tokens, token_to_idx, max_token_length):
  encoded = np.zeros(max_token_length, dtype=np.int32)
  for i, token in enumerate(tokens):
    if token in token_to_idx:
      encoded[i] = token_to_idx[token]
    else:
      encoded[i] = token_to_idx['<UNK>']
  return encoded


def encode_captions(data, token_to_idx, max_token_length):
  encoded_list = []
  parts_list = [] #--!!!
  lengths = []
  for img in data:
    for region in img['relationships']:
      tokens = region['tokens']
      if tokens is None: continue
      tokens_encoded = encode_caption(tokens, token_to_idx, max_token_length)
      encoded_list.append(tokens_encoded)
      parts_list.append(encode_caption(region['parts'], {1:1,2:2,3:3}, max_token_length))#---!!!)
      lengths.append(len(tokens))
  return np.vstack(encoded_list), np.vstack(parts_list) , np.asarray(lengths, dtype=np.int32)#--!!!


def encode_boxes(data, original_heights, original_widths, image_size):
  all_boxes = []
  xwasbad = 0
  ywasbad = 0
  wwasbad = 0
  hwasbad = 0
  for i, img in enumerate(data):
    H, W = original_heights[i], original_widths[i]
    scale = float(image_size) / max(H, W)
    for region in img['relationships']:
      if region['tokens'] is None: continue
      # recall: x,y are 1-indexed
      
      x, y = round(scale*(region['subject']['x']-1)+1), round(scale*(region['subject']['y']-1)+1)
      w, h = round(scale*region['subject']['w']), round(scale*region['subject']['h'])  
      #----------------------------------------------subject-----------------------------------------------------
      # clamp to image
      if x < 1: x = 1
      if y < 1: y = 1
      if x > image_size - 1: 
        x = image_size - 1
        xwasbad += 1
      if y > image_size - 1: 
        y = image_size - 1
        ywasbad += 1
      if x + w > image_size: 
        w = image_size - x
        wwasbad += 1
      if y + h > image_size: 
        h = image_size - y
        hwasbad += 1

      box = np.asarray([x+floor(w/2), y+floor(h/2), w, h], dtype=np.int32) # also convert to center-coord oriented
      assert box[2]>=0 # width height should be positive numbers
      assert box[3]>=0
      all_boxes.append(box)
      
      #------------------------------------------------object--------------------------------------------------------
      x, y = round(scale*(region['object']['x']-1)+1), round(scale*(region['object']['y']-1)+1)
      w, h = round(scale*region['object']['w']), round(scale*region['object']['h'])  
      
      # clamp to image
      if x < 1: x = 1
      if y < 1: y = 1
      if x > image_size - 1: 
        x = image_size - 1
        xwasbad += 1
      if y > image_size - 1: 
        y = image_size - 1
        ywasbad += 1
      if x + w > image_size: 
        w = image_size - x
        wwasbad += 1
      if y + h > image_size: 
        h = image_size - y
        hwasbad += 1

      box = np.asarray([x+floor(w/2), y+floor(h/2), w, h], dtype=np.int32) # also convert to center-coord oriented
      assert box[2]>=0 # width height should be positive numbers
      assert box[3]>=0
      all_boxes.append(box)
  print 'number of bad x,y,w,h: ', xwasbad, ywasbad, wwasbad, hwasbad
  
  return np.vstack(all_boxes)

def build_img_idx_to_box_idxs(data):
  img_idx = 1
  box_idx = 1
  num_images = len(data)
  img_to_first_box = np.zeros(num_images, dtype=np.int32)
  img_to_last_box = np.zeros(num_images, dtype=np.int32)
  for img in data:
    img_to_first_box[img_idx - 1] = box_idx
    for region in img['relationships']:
      if region['tokens'] is None: continue
      box_idx += 2#-----------------------------!!!!!!!!!!! add 2 box for 1 image !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    img_to_last_box[img_idx - 1] = box_idx - 1 # -1 to make these inclusive limits
    img_idx += 1
  
  
  return img_to_first_box, img_to_last_box

def build_filename_dict(data):
  # First make sure all filenames
  filenames_list = [img['image_id'] for img in data]
  assert len(filenames_list) == len(set(filenames_list))
  
  next_idx = 1
  filename_to_idx, idx_to_filename = {}, {}
  for img in data:
    filename = img['image_id']
    filename_to_idx[filename] = next_idx
    idx_to_filename[next_idx] = filename
    next_idx += 1
  return filename_to_idx, idx_to_filename

def encode_filenames(data, filename_to_idx):
  filename_idxs = []
  for img in data:
    filename = img['image_id']
    idx = filename_to_idx[filename]
    for region in img['relationships']:
      if region['tokens'] is None: continue
      filename_idxs.append(idx)#do it twice, or not?
  
  return np.asarray(filename_idxs, dtype=np.int32)

def add_images(data, h5_file, args):
  num_images = len(data)
  
  shape = (num_images, 3, args.image_size, args.image_size)
  image_dset = h5_file.create_dataset('images', shape, dtype=np.uint8)
  original_heights = np.zeros(num_images, dtype=np.int32)
  original_widths = np.zeros(num_images, dtype=np.int32)
  image_heights = np.zeros(num_images, dtype=np.int32)
  image_widths = np.zeros(num_images, dtype=np.int32)
  
  lock = Lock()
  q = Queue()
  
  for i, img in enumerate(data):
    if i < 3780:
      filename = os.path.join(args.train_image_dir, img['image_id'])
    else:
      filename = os.path.join(args.test_image_dir, img['image_id'])
    q.put((i, filename))
    
  def worker():
    while True:
      i, filename = q.get()
      img = imread(filename)
      # handle grayscale
      if img.ndim == 2:
        img = img[:, :, None][:, :, [0, 0, 0]]
      H0, W0 = img.shape[0], img.shape[1]
      img = imresize(img, float(args.image_size) / max(H0, W0))
      H, W = img.shape[0], img.shape[1]
      # swap rgb to bgr. Is this the best way?
      r = img[:,:,0].copy()
      img[:,:,0] = img[:,:,2]
      img[:,:,2] = r

      lock.acquire()
      if i % 1000 == 0:
        print 'Writing image %d / %d' % (i, len(data))
      original_heights[i] = H0
      original_widths[i] = W0
      image_heights[i] = H
      image_widths[i] = W
      image_dset[i, :, :H, :W] = img.transpose(2, 0, 1)
      lock.release()
      q.task_done()
  
  print('adding images to hdf5.... (this might take a while)')
  for i in xrange(args.num_workers):
    t = Thread(target=worker)
    t.daemon = True
    t.start()
  q.join()

  h5_file.create_dataset('image_heights', data=image_heights)
  h5_file.create_dataset('image_widths', data=image_widths)
  h5_file.create_dataset('original_heights', data=original_heights)
  h5_file.create_dataset('original_widths', data=original_widths)

def words_preprocess(phrase):
  """ preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
  replacements = {
    u'½': u'half',
    u'—' : u'-',
    u'™': u'',
    u'¢': u'cent',
    u'ç': u'c',
    u'û': u'u',
    u'é': u'e',
    u'°': u' degree',
    u'è': u'e',
    u'…': u'',
  }
  for k, v in replacements.iteritems():
    phrase = phrase.replace(k, v)
  #pdb.set_trace()
  return str(phrase.encode("utf-8")).lower().translate(None, string.punctuation).split()

def split_filter_captions(data, max_token_length, tokens_type, verbose=True):
  """
  Modifies data in-place by adding a 'tokens' field to each region.
  If the region's label is too long, 'tokens' will be None; otherwise
  it will be a list of strings.
  Splits by space when tokens_type = "words", or lists all chars when "chars"
  """
  captions_kept = 0
  captions_removed = 0
  for i, img in enumerate(data):
    if verbose and (i + 1) % 2000 == 0:
      print 'Splitting tokens in image %d / %d' % (i + 1, len(data))
    regions_per_image = 0
    img_kept, img_removed = 0, 0
    for region in img['relationships']:
      # create tokens array
      if tokens_type == 'words':
        tokens = words_preprocess(region['phrase'])
      elif tokens_type == 'chars':
        tokens = list(region['label'])
      else:
        assert False, 'tokens_type must be "words" or "chars"'

      
      # filter by length
      if max_token_length > 0 and len(tokens) <= max_token_length:
        
        region['tokens'] = tokens
        #pdb.set_trace()
        region['parts'] = [1] * len(words_preprocess(region['phrase_subj'])) + [2] * len(words_preprocess(region['phrase_pred'])) + [3] * len(words_preprocess(region['phrase_obj']))#--!!!
        
        captions_kept += 1
        img_kept += 1
        regions_per_image = regions_per_image + 1
      else:
        region['tokens'] = None
        captions_removed += 1
        img_removed += 1
    
    if regions_per_image == 0:
      #pdb.set_trace()
      print 'kept %d, removed %d' % (img_kept, img_removed)
      assert False, 'DANGER, some image has no valid regions. Not super sure this doesnt cause bugs. Think about more if it comes up'

  if verbose:
    print 'Keeping %d captions' % captions_kept
    print 'Skipped %d captions for being too long' % captions_removed

def encode_splits(data, split_data):
  """ Encode splits as intetgers and return the array. """
  lookup = {'train': 0, 'val': 1, 'test': 2}
  id_to_split = {}
  split_array = np.zeros(len(data))
  for split, idxs in split_data.iteritems():
    for idx in idxs:
      id_to_split[idx] = split
  for i, img in enumerate(data):
    split_array[i] = lookup[id_to_split[img['image_id']]]
  return split_array


def VRD_to_VGR(data, input_data,objects,predicates):
  data_list = input_data.keys()
  save_list = []# collect the list who has nonzero relationships
  
  for i, img in enumerate(input_data):#images 
    data_tmp = {}
    data_tmp['image_id'] = data_list[i]
    data_tmp['relationships']=[]
    
    if len(input_data[data_list[i]]) == 0:#if no relationship label, skip
      continue
    
    save_list.append(i)
    for relationship in  input_data[data_list[i]]:#relationships in image
      tmp={}
      tmp['predicate'] = relationship['predicate']

      tmp['object']={}
      tmp['object']['name'] = relationship['object']['category']
      tmp['object']['h'] = relationship['object']['bbox'][1]-relationship['object']['bbox'][0]
      tmp['object']['w'] = relationship['object']['bbox'][3]-relationship['object']['bbox'][2]
      tmp['object']['y'] = relationship['object']['bbox'][0]
      tmp['object']['x'] = relationship['object']['bbox'][2]
      
      tmp['subject']={}
      tmp['subject']['name'] = relationship['subject']['category']
      tmp['subject']['h'] = relationship['subject']['bbox'][1]-relationship['subject']['bbox'][0]
      tmp['subject']['w'] = relationship['subject']['bbox'][3]-relationship['subject']['bbox'][2]
      tmp['subject']['y'] = relationship['subject']['bbox'][0]
      tmp['subject']['x'] = relationship['subject']['bbox'][2]
      
      #pdb.set_trace()      
      phrase = objects[relationship['subject']['category']]+' '+ predicates[relationship['predicate']]+' '+objects[relationship['object']['category']]

      tmp['phrase'] = phrase
      tmp['phrase_subj'] = objects[relationship['subject']['category']]#--!!!
      tmp['phrase_pred'] = predicates[relationship['predicate']]#--!!!
      tmp['phrase_obj'] = objects[relationship['object']['category']]#--!!!
    
      data_tmp['relationships'].append(tmp)

    data.append(data_tmp)
  
  data_list = [data_list[k] for k in save_list]
  return data,data_list

def main(args):

  # read in the data
  with open(args.train_data, 'r') as f:
    train_data = json.load(f)
  with open(args.test_data, 'r') as f:
    test_data = json.load(f)
  
  with open(args.predicate, 'r') as f:
    predicates = json.load(f)
  with open(args.object, 'r') as f:
    objects = json.load(f)
  
  
  data,train_list = VRD_to_VGR([],train_data,objects,predicates)
  
  data,test_list = VRD_to_VGR(data,test_data,objects,predicates)
  
  
  split_data = {'train':train_list, 'val':test_list}
    
    #split_data = json.load(f)
  
  # Only keep images that are in a split
  print 'There are %d images total' % len(data)
  
  
  # create the output hdf5 file handle
  f = h5py.File(args.h5_output, 'w')

  # add several fields to the file: images, and the original/resized widths/heights
  add_images(data, f, args)

  # add split information
  split = encode_splits(data, split_data)
  f.create_dataset('split', data=split)

  # process "label" field in each region to a "tokens" field, and cap at some max length
  split_filter_captions(data, args.max_token_length, args.tokens_type)

  # build vocabulary
  vocab,vocab_pred,vocab_obj = build_vocab(data, args.min_token_instances) # vocab is a set()
  token_to_idx, idx_to_token = build_vocab_dict(vocab) # both mappings are dicts
    
  # encode labels
  captions_matrix, parts_matrix, lengths_vector = encode_captions(data, token_to_idx, args.max_token_length)#--!!!
  #pdb.set_trace()
  f.create_dataset('labels', data=np.concatenate((captions_matrix,parts_matrix),axis=1))#--!!!
  f.create_dataset('lengths', data=lengths_vector)
  
  # encode boxes
  original_heights = np.asarray(f['original_heights'])
  original_widths = np.asarray(f['original_widths'])
  boxes_matrix = encode_boxes(data, original_heights, original_widths, args.image_size)
  
  f.create_dataset('boxes', data=boxes_matrix)
  
  # integer mapping between image ids and box ids
  img_to_first_box, img_to_last_box = build_img_idx_to_box_idxs(data)
  f.create_dataset('img_to_first_box', data=img_to_first_box)
  f.create_dataset('img_to_last_box', data=img_to_last_box)
  filename_to_idx, idx_to_filename = build_filename_dict(data)
  box_to_img = encode_filenames(data, filename_to_idx)
  
  f.create_dataset('box_to_img', data=box_to_img)
  f.close()

  # and write the additional json file 
  json_struct = {
    'token_to_idx': token_to_idx,
    'idx_to_token': idx_to_token,
    'filename_to_idx': filename_to_idx,
    'idx_to_filename': idx_to_filename,
  }
  with open(args.json_output, 'w') as f:
    json.dump(json_struct, f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # INPUT settings
  parser.add_argument('--train_data',
      default='data/VRD/annotations_train.json',
      help='Input JSON file with train relationships')
  parser.add_argument('--test_data',
      default='data/VRD/annotations_test.json',
      help='Input JSON file with test relationships')
  
  parser.add_argument('--predicate',
      default='data/VRD/predicates.json',
      help='Input JSON file predicates label')
  parser.add_argument('--object',
      default='data/VRD/objects.json',
      help='Input JSON file objects label')
      
  parser.add_argument('--train_image_dir',
      default='data/VRD/sg_dataset/sg_train_images',
      help='Directory containing train images')
  parser.add_argument('--test_image_dir',
      default='data/VRD/sg_dataset/sg_test_images',
      help='Directory containing test images')

  # OUTPUT settings
  parser.add_argument('--json_output',
      default='data/VG-regions-dicts_VRD2.json',
      help='Path to output JSON file')
  parser.add_argument('--h5_output',
      default='data/VG-regions_VRD2.h5',
      help='Path to output HDF5 file')

  # OPTIONS
  parser.add_argument('--image_size',
      default=720, type=int,
      help='Size of longest edge of preprocessed images')  
  parser.add_argument('--max_token_length',
      default=15, type=int,
      help="Set to 0 to disable filtering")
  parser.add_argument('--min_token_instances',
      default=3, type=int,#--originally 15
      help="When token appears less than this times it will be mapped to <UNK>")
  parser.add_argument('--tokens_type', default='words',
      help="Words|chars for word or char split in captions")
  parser.add_argument('--num_workers', default=5, type=int)
  parser.add_argument('--max_images', default=-1, type=int,
      help="Set to a positive number to limit the number of images we process")
  args = parser.parse_args()
  main(args)

