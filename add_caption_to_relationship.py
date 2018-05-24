import argparse, os, json, string
import numpy as np

import pdb

#import sys
#sys.path.insert(0,'order-embedding')
#import tools
#model = tools.load_model('order-embedding/snapshots/order')


def match_caption_with_relationship(region_data, relationship_data):
  """ find the closest caption for each relationship and match them""" 
  MAX_L=0
  for iid , img in enumerate(relationship_data):
    print 'matching caption for relatoinship of img #%d/%d'%(iid,len(relationship_data))
    sentence=[]
    #for region in region_data[iid]['regions']:
    #    sentence.append(region['phrase'])
    #candidate = tools.encode_sentences(model,sentence,verbose=False)#sentence embedding
    for rid, relationship in enumerate(img['relationships']):
        
        phrase = relationship['subject']['name']+' '+ relationship['predicate']+' '+relationship['object']['name']
        #query = tools.encode_sentences(model,[phrase],verbose=False)#sentence embedding
 
        #err=-np.dot(candidate,query.T)#measuring distances with all sentences
        
        relationship_data[iid]['relationships'][rid]['phrase'] = phrase#sentence[err.argmin()]#predicted index---!!!
        if MAX_L < len(phrase.split()):
          MAX_L = len(phrase.split())
          MAX_phrase =[relationship['subject']['name'],relationship['predicate'],relationship['object']['name']]
          
    
  pdb.set_trace() 
  return relationship_data
  
  
def main(args):  
  # read in the data
  with open(args.region_data, 'r') as f:
    region_data = json.load(f)
  with open(args.relationship_data, 'r') as f:
    data = json.load(f)
  
  # Only keep images that are in a split
  print 'There are %d images total' % len(region_data)

  #find and match closest caption for each relationship
  data = match_caption_with_relationship(region_data, data)
  
  with open(args.json_output, 'w') as f:
    json.dump(data, f)



  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # INPUT settings
  parser.add_argument('--region_data',
      default='data/visual-genome/1.2/region_descriptions.json',
      help='Input JSON file with regions and captions')
  parser.add_argument('--relationship_data',
      default='data/visual-genome/1.2/relationships_with_caption.json',
      help='Input JSON file with relationships')


  # OUTPUT settings
  parser.add_argument('--json_output',
      default='data/visual-genome/1.2/relationships_as_caption.json',
      help='Path to output JSON file')


  # OPTIONS
  parser.add_argument('--image_size',
      default=720, type=int,
      help='Size of longest edge of preprocessed images')  
  parser.add_argument('--max_token_length',
      default=15, type=int,
      help="Set to 0 to disable filtering")


  args = parser.parse_args()
  main(args)
