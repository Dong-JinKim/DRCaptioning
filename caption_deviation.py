# coding=utf8

import argparse, os, json, string

from math import floor
import h5py
import numpy as np
from scipy.misc import imread, imresize

import pdb

import numpy as np

import sys
sys.path.insert(0,'order-embedding')
import tools
model = tools.load_model('order-embedding/snapshots/order')



def encode_caption(tokens, token_to_idx, max_token_length):
  encoded = np.zeros(max_token_length, dtype=np.int32)
  for i, token in enumerate(tokens):
    if token in token_to_idx:
      encoded[i] = token_to_idx[token]
    else:
      encoded[i] = token_to_idx['<UNK>']
  return encoded
  
def main(args):

  # read in the data
  with open(args.result_path, 'r') as f:
    data = json.load(f)
  with open(args.data_json, 'r') as f:
    vocab = json.load(f)
  
  options = data['opt']
  captions = data['captions']
  vocab_size = options['vocab_size']
  token_to_idx = vocab['token_to_idx']
  idx_to_token = vocab['idx_to_token']
  
  

  deviation = [0]*len(captions) 
  all_vecs = np.zeros((0,1024),'float32')
  
  for iid, img in enumerate(captions):
    print 'collecting captions (%d/%d)'%((iid),len(captions))
    num_of_box = int((np.sqrt(4*len(img)+1)+1)/2)
    
    per_img_hist = np.array([0]*vocab_size)#hist among imgs
    per_box_hist = np.array(  [([0]*vocab_size)]*num_of_box    )
    words_per_box = [0]*num_of_box
    

        
    #pdb.set_trace()
    vectors = tools.encode_sentences(model,img,verbose=False)#sentence embedding
    #pdb.set_trace()
    all_vecs = np.concatenate((all_vecs, vectors),axis=0)
    deviation[iid] = np.std(np.array(vectors))

    #pdb.set_trace()
  
  #captions
  mean_deviation = np.mean(deviation)
  

  


  print 'mean stanard deviation=%.3f'%(mean_deviation)
  print 'total stanard deviation=%.3f'%np.std(np.array(all_vecs))
  pdb.set_trace()
  return mean_deviation


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--result_path',
      default='relcap_statistics_union75.json',
      help='The JSON file to with caption resutls.')
  parser.add_argument('--data_json',
      default='data/VG-regions-dicts_R.json',
      help='The JSON file to load data from; optional.')    
  args = parser.parse_args()
  main(args)

