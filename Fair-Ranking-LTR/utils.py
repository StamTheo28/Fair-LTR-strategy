import pandas as pd

def get_document_direct(index, docno=None, docid=None):
  if docid is None and docno is None:
    raise ValueError("Must specify docno or docid")
  if docno is not None:
    docid = index.getMetaIndex().getDocument("docno", docno)
  else:
    docno = index.getMetaIndex().getItem("docno", docid)
  rtr = "Docno %s (docid %d)\n" % (docno, docid)
  pointer = index.getDocumentIndex().getDocumentEntry(docid)
  for p in index.getDirectIndex().getPostings(pointer):
    rtr += p.toString()
   #termid = p.getId()
    #term = index.getLexicon()[termid].getKey()
    #rtr += ("\t%s %d\n" % ( term, p.getFrequency()))
  return rtr

def get_doc(docid):
    corpus_path = "/media/sf_Trec-fair-2022/trec_2022_articles_discrete.json/trec_2022_articles_discrete_V2.json"
    data = pd.read_json(corpus_path, orient='records', lines = True, chunksize=100e0000)
    count = 0
    for i in data:   
        print(count)
        count +=1
        res = i[i['page_id'].isin([docid])]
        if len(res) != 0:
            print('yas')
            return i[i['page_id'] == docid]
        else:
            continue
        
    return 
