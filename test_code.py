import json
import pdb
import numpy as np

split = json.load(open('info/densecap_splits.json'))
data = json.load(open('data/visual-genome/1.0/relationships_as_caption.json'))



# np.where(np.array([img['image_id'] for img in data])==713715)
# [rel['phrase'] for rel in data[1597]['relationships']]

test_list = []
phrase = []
subj=[]
obj=[]

pdb.set_trace()
for iid, img in enumerate(data):
  print("searching...(%d/%d)"%(iid, len(data)))
  for relationships in img['relationships']:
    #pdb.set_trace()
    if ('dog' in relationships['subject']['name'].split() or 'dog' in relationships['object']['name'].split())and not(relationships['object']['name']=='hot dog') and not(relationships['object']['name']=='hot dog') and len(relationships['phrase'].split())>4:
    #if 'horse' in relationships['object']['name'].split() and 'riding' in relationships['predicate'].split() and len(relationships['phrase'].split())==5:
    #if len(relationships['phrase'].split())==5 and len(relationships['subject']['name'].split())*len(relationships['object']['name'].split())*len(relationships['predicate'].split())>3 :#relationships['subject']['name']=='little boy':# and  relationships['predicate'] == 'WEARING' :
      test_list.append(img['image_id'])
      phrase.append(relationships['phrase'])
      subj.append(relationships['subject'])
      obj.append(relationships['object'])
      continue
  
pdb.set_trace()  
with open('dog.txt','w') as f:
  f.writelines("%d\n"%aaa for aaa in test_list)
with open('dog_phrase.txt','w') as f:
  f.writelines("%s_\n"%aaa for aaa in phrase) 
  
  
  
  #if img['image_id'] in split['test']:











