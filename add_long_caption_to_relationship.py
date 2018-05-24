import argparse, os, json, string
import numpy as np

import pdb
import nltk


def compute_iou(spatial,target):
  
  area1 = spatial['w']*spatial['h']
  area2 = target['w']*target['h']

  x0 = max(spatial['x'],target['x'])
  y0 = max(spatial['y'],target['y'])
  x1 = min(spatial['x']+spatial['w'],target['x']+target['w'])
  y1 = min(spatial['y']+spatial['h'],target['y']+target['h'])
  
  w = max( x1-x0, 0 )
  h = max( y1-y0, 0 )
  
  intersection = w*h
  
  
  if area1-intersection+area2 ==0:
    iou = 0#label is too noisy... zero area...
  else:
    iou = float(intersection)/(area1-intersection+area2)
  
  return iou

def select_best_word(candidates ,  target):#compare spatial relationships and select best word for target region
  IOUs = []
  for cid, spatial in enumerate(candidates):
    iou = compute_iou(spatial,target)
    if iou>0.3:# add only if IOU is matched enough
      IOUs.append(iou)
  #pdb.set_trace()
  if len(IOUs)==0:# if all samples are rejected due to IOU
    return []#-1
  #elif len(IOUs)==1:
  #  return IOUs.index(max(IOUs))#---!!!!
  else:
    return [ind for ind,val in enumerate(IOUs) if val==max(IOUs)]
  
  
def match_caption_with_relationship(region_data, attribute_data, relationship_data):
  """ find the closest caption for each relationship and match them""" 
  MAX_L=0
  for iid , img in enumerate(relationship_data):
    print 'matching caption for relatoinship of img #%d/%d'%(iid,len(relationship_data))
    
    #pdb.set_trace()
    
    #filter out only objects that has attributes
    obj_with_att = [att for att in attribute_data[iid]['attributes'] if ('attributes' in att.keys())]

    ##names of objects
    #object_names = [at for att in obj_with_att for at in att['names']*len(att['attributes'])]
    ## attributes for corresponding objects
    #attributes = [at   for att in obj_with_att for at in att['attributes'] ]    
    ## spatial information for attributes
    #spatials = [at for att in obj_with_att for at in [{'x':att['x'],'y':att['y'],'w':att['w'],'h':att['h']}]*len(att['attributes'])]
    
    
    object_names = [att['names'] for att in obj_with_att ]
    attributes = [att['attributes'] for att in obj_with_att]    
    spatials = [{'x':att['x'],'y':att['y'],'w':att['w'],'h':att['h']} for att in obj_with_att ]
    
    
    for rid, relationship in enumerate(img['relationships']):
        
        
        #pdb.set_trace()
        
        # matching indexs for subject (box index that has matched attribute)
        subj_box = np.where([relationship['subject']['name'] in ss for ss in object_names])[0]
        # matching indexs for object  (box index that has matched attribute)
        obj_box = np.where([relationship['object']['name'] in ss for ss in object_names])[0]
        
        
        #[attributes[ind] for ind in subj_box]
        
        basic_phrase = relationship['subject']['name']+' '+ relationship['predicate']+' '+ relationship['object']['name']# only S-V-O
        
        
        #-----------------------------------------------------------------------
        #----------------subj---------------------------------------------------
        #pdb.set_trace()
        #subj_box = subj_box[np.where(  [attributes[ind] not in basic_phrase for ind in subj_box]  )]
        
        
        #[atts for ind in subj_box for atts in attributes[ind] if atts not in basic_phrase]
        
        #[[atts  for atts in attributes[ind] if atts not in basic_phrase] for ind in subj_box ]
        
        
        
        if len(subj_box)!=0:
          #select best IOU
          selected  = select_best_word([spatials[ind] for ind in subj_box], relationship['subject'])
          selected_words = [atts for ss in selected for atts in attributes[subj_box[ss]]]
          
          #filter out the words that has been appear in original sentence
          #pdb.set_trace()
          selected_words = [word for word in selected_words if (word not in basic_phrase and relationship['subject']['name'] not in word)]
          
          POS = nltk.pos_tag(selected_words)#check POS and filter out
          #print(POS)#pdb.set_trace()
          selected_words = [pp[0] for pp in POS if pp[1] in ['VBN','JJ','NN','VBG','VBD']]
          
          if len(selected_words) !=0:# all attributes rejected
            #if len(selected_words)>1:
              #pdb.set_trace()
            #pdb.set_trace()
            selected_subj_att  = selected_words[rid%len(selected_words)] + ' '
          else:
            selected_subj_att = ''
        else:
          selected_subj_att = ''
        
        if selected_subj_att == '':#if noting to add try to add 'the'
          first_word = relationship['subject']['name'].split()[0]
          if first_word=='the' or first_word=='a':
            selected_subj_att = ''
          else:# if therse no 'a' or 'the' add the
            selected_subj_att = 'the ' #'the '---!!!!
            
        #-----------------------------------------------------------------------
        #----------------obj----------------------------------------------------
        
        #filter out the words that has been appear in original sentence
        #obj_box = obj_box[np.where([attributes[ind] not in basic_phrase for ind in obj_box])]
        
        if len(obj_box)!=0:
          #select best IOU
          selected  = select_best_word([spatials[ind] for ind in obj_box], relationship['object'])
          selected_words = [atts for ss in selected for atts in attributes[obj_box[ss]]]
          
          #filter out the words that has been appear in original sentence
          selected_words = [word for word in selected_words if (word not in basic_phrase and relationship['object']['name'] not in word)]

          POS = nltk.pos_tag(selected_words)#check POS and filter out
          #print(POS)#pdb.set_trace()
          selected_words = [pp[0] for pp in POS if pp[1] in ['VBN','JJ','NN','VBG','VBD']]
          
          if len(selected_words) !=0:# all attributes rejected
            #if len(selected_words)>1:
              #pdb.set_trace()
            selected_obj_att  = selected_words[rid%len(selected_words)] + ' '
          else:
            selected_obj_att = ''
        else:
          selected_obj_att = ''
        
        
        #--------------------------------------------------------------------------
        
        
        
        relationship['subject']['name'] = selected_subj_att  + relationship['subject']['name']
        relationship['object']['name'] = selected_obj_att + relationship['object']['name']
        
        
        phrase = relationship['subject']['name']+' '+ relationship['predicate']+' '+ relationship['object']['name']
        
        print phrase
        
        #pdb.set_trace()
        
        relationship_data[iid]['relationships'][rid]['phrase'] = phrase

          
    
  #pdb.set_trace() 
  return relationship_data
  
  
def main(args):  
  # read in the data
  with open(args.region_data, 'r') as f:
    region_data = json.load(f)
  with open(args.attribute_data, 'r') as f:
    attribute_data = json.load(f)
  with open(args.relationship_data, 'r') as f:
    data = json.load(f)
  
  # Only keep images that are in a split
  print 'There are %d images total' % len(region_data)

  #find and match closest caption for each relationship
  data = match_caption_with_relationship(region_data, attribute_data, data)
  
  
  #pdb.set_trace()
  #len([1  for img in data for rel in img['relationships'] if len(rel['phrase'].split())==15])
  with open(args.json_output, 'w') as f:
    json.dump(data, f)



  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # INPUT settings
  parser.add_argument('--region_data',
      default='data/visual-genome/1.2/region_descriptions.json',
      help='Input JSON file with regions and captions')
  parser.add_argument('--relationship_data',
      default='data/visual-genome/1.2/relationships.json',
      help='Input JSON file with relationships')
  parser.add_argument('--attribute_data',
      default='data/visual-genome/1.2/attributes.json',
      help='Input JSON file with relationships')

  # OUTPUT settings
  parser.add_argument('--json_output',
      default='data/visual-genome/1.2/relationships_as_long_caption4.json',
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
