# coding=utf8

import argparse, os, json, string

from math import floor
import h5py
import numpy as np
from scipy.misc import imread, imresize

import pdb

import numpy as np


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
    #u'\xf1':u'',
    #u'\xff':u'',
    #u'\xea':u'',
  }
  for k, v in replacements.iteritems():
    phrase = phrase.replace(k, v)
  #pdb.set_trace()
  return str(phrase.encode("utf-8")).lower().translate(None, string.punctuation).split()

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
  vocab_size = int(options['vocab_size'])
  token_to_idx = vocab['token_to_idx']
  idx_to_token = vocab['idx_to_token']
  
  
  #with open(args.result_path, 'r') as f:
  #  captions = json.load(f)
  #pdb.set_trace()
  total_hist = np.array([0]*vocab_size)#hist among all data
  
  per_box_hist_tot = np.array([0]*vocab_size)#hist among boxes
  
  words_per_img = [0]*len(captions) 
  words_per_box_tot = []
  #pdb.set_trace()
  for iid, img in enumerate(captions):# all images
    print 'collecting captions (%d/%d)'%((iid),len(captions))
    num_of_box = int((np.sqrt(4*len(img)+1)+1)/2)
    
    per_img_hist = np.array([0]*vocab_size)#hist among imgs
    per_box_hist = np.array(  [([0]*vocab_size)]*num_of_box    )
    words_per_box = [0]*num_of_box
    
    for bid in range(num_of_box):
      for bjd in range(num_of_box):
        
        cid = bid*(num_of_box-1)+bjd
        if bid==bjd:
          continue

        cap = img[cid]
    

        #per_box_hist = np.array([0]*vocab_size)
        encoded = encode_caption(words_preprocess(cap),token_to_idx,15 )
        hist_tmp,_=np.histogram(encoded,bins=range(vocab_size+1))
        
        total_hist = total_hist + hist_tmp
        per_box_hist[bid] = per_box_hist[bid]  + hist_tmp
        per_box_hist[bjd] = per_box_hist[bjd]  + hist_tmp
        per_img_hist = per_img_hist + hist_tmp
        
        len(np.where(per_img_hist[1:]>0)[0])
        
    #pdb.set_trace()    
    

    words_per_box = [len(np.where(per_box_hist[bid][1:]>0)[0]) for bid in range(num_of_box)]
    
    words_per_box_tot = words_per_box_tot + words_per_box
    words_per_img[iid] = len(np.where(per_img_hist[1:]>0)[0])
    #pdb.set_trace()
  
  #captions
  total_words = len(np.where(total_hist[1:]>0)[0])
  

  

  pdb.set_trace()
  print 'total vocab =%d, words-per-img=%.3f, words-per-box=%.3f'%(total_words, np.mean(words_per_img),np.mean(words_per_box_tot))
  
  #pdb.set_trace()
  return total_words


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--result_path',
      default='relcap_statistics_MTTS4_75.json',
      help='The JSON file to with caption resutls.')
  parser.add_argument('--data_json',
      default='data/VG-regions-dicts_R2longv3.json',
      help='The JSON file to load data from; optional.')    
  args = parser.parse_args()
  main(args)

