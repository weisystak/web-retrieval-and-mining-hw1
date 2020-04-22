import xml.etree.ElementTree as ET
import os
import math
import numpy as np
from numpy import dot
from numpy.linalg import norm as nm
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
import scipy.sparse
import rocchio as ro
import argparse

parser = argparse.ArgumentParser(description='vsm')
parser.add_argument('-r', action='store_true', dest="r")
parser.add_argument('-i', type=str, dest="queryFile")
parser.add_argument('-o', type=str, dest="rankList")
parser.add_argument('-m', type=str, dest="modelDir")
parser.add_argument('-d', type=str, dest="docDir")
args = parser.parse_args()


queryDir = "queries"
docDir = args.docDir   # "/CIRB010"
# modelDir = "model"
modelDir = args.modelDir

# dataset = "train" 
# dataset = "test"

# queryFile = queryDir + "/query-" + dataset + ".xml"
queryFile = args.queryFile

k = 1.5
b = 0.75


rocchio = ""
if args.r:
    rocchio = "rocchio"

# ansFile = "ans_" + dataset + "_k_" + str(k) + "_b_" + str(b) + "_" + rocchio + ".csv"
ansFile = args.rankList


doc_list = open(modelDir + "/file-list", "r").readlines()
vocab = open(modelDir+"/vocab.all", "r").readlines()
ifile = open(modelDir+"/inverted-file", "r")

n_docs = len(doc_list)
n_vocs = len(vocab)

vocs = np.empty(n_vocs, dtype=object)
ngrams = {}
docIDs = np.empty(n_docs, dtype=object)
docID2idx = {}


def normalize_tf( tf, idf, lens, avgLen):
    if type(lens) == np.matrix:
        tf = tf.tocoo()
        # tf = tf * (k+1) / ( tf.data + k*(1-b+b*lens/avgLen)[:, np.newaxis] )
        t1 = tf*(k+1)
        t2 = k*(1-b+b*lens/avgLen)
        tf.data += np.array(t2[tf.row]).reshape(len(tf.data),)
        tf.data = t1.data/tf.data
        tf.data *= idf[tf.col]
        tf = tf.tocsr()
        return  tf
    else:
        return ( tf * (k+1) / ( tf + k*(1-b+b*lens/avgLen) ) ) * idf



# construct vocs
for i, line in enumerate(vocab):
    vocs[i] = line.rstrip()

# compute tf idf using ifile

row = []
col = []
data = []

c=0
for line in tqdm(ifile, desc="ifile"):
    a = [ int(x) for x in line.split() ]
    voc1 = vocs[a[0]]
    
    if a[1] == -1: # unigram
        voc2 = ""
    else:          # bigram
        voc2 = " " + vocs[a[1]]
    dfreq = a[2]

    if voc1+voc2 in ngrams:
        print(voc1+voc2)
        print(a)
        for i in range(dfreq):
            ifile.readline()
    else:
        ngrams[ voc1+voc2 ] = { "doc freq": dfreq, "vid":  c, "docs": {} }
        for i in range(dfreq):
            line = ifile.readline()
            doc, tf = line.split()

            row.append(int(doc))
            col.append(c)
            data.append(int(tf))
        c += 1
        # ngrams[ voc1+voc2 ]["docs"][int(doc)] = int(tf)

dim = len(ngrams)


# parse docs
# construct docIDs
for i, docName in enumerate(doc_list):
    
    root = ET.parse(docDir + "/" + docName.strip() ).getroot()
    docIDs[i] =  root[0][0].text # root.find("id").text
    docID2idx[docIDs[i]] = i

    title = root[0].find("title").text #root[0][2].text
    # print("title: " + title)  
    if title is None:
        continue 
    for j in range(len(title)-1):
        c = title[j] + " " + title[j+1]
        if  c in ngrams:
            vid = ngrams[c]["vid"]
            
            row.append(i)
            col.append(vid)
            data.append(8)
    

idf = np.empty(dim)
doc_tf =  csr_matrix((data, (row, col)), shape=(n_docs, dim), dtype='float') # np.zeros((n_docs, dim))

for vinfo in ngrams.values():
    vid = vinfo["vid"]
    idf[vid] = math.log( ( n_docs - vinfo["doc freq"] + 0.5) / (vinfo["doc freq"] + 0.5) )
    # for doc, tf in vinfo["docs"].items():
    #     doc_tf[doc][vid] = tf

docLens = doc_tf.sum(1)
avgLen = docLens.mean()

doc_vecs = normalize_tf(doc_tf, idf, docLens, avgLen)


# print(avgLen)


print("save vecs...")

# scipy.sparse.save_npz('doc_vecs.npz', doc_vecs)
# np.save("idf.npy", idf)
# doc_vecs = scipy.sparse.load_npz('doc_vecs.npz')
# idf = np.load("idf.npy")

# print(doc_vecs[0])
# print(idf)

# avgLen = 1378.8242144256153
# dim = 1193467

out = open( ansFile, "w")
out.write("query_id,retrieved_docs\n")


root = ET.parse(queryFile).getroot()
w = 5
for topic in root.findall('topic'):
    qVec = np.zeros(dim)

    num = topic[0].text[-3:]
    title = topic[1].text
    question = topic[2].text
    narrative = topic[3].text
    concepts = topic[4].text.rstrip()
    a = title+question+concepts
    x = narrative
    qLen = len(title+question+concepts) * w + len(narrative)


    for c in a:
        if c in ngrams:
            vid = ngrams[c]["vid"]
            qVec[vid] += w
    for c in x:
        if c in ngrams:
            vid = ngrams[c]["vid"]
            qVec[vid] += 1
    for i in range(len(a)-1):
        c = a[i]+ " " + a[i+1]
        if  c in ngrams:
            vid = ngrams[c]["vid"]
            qVec[vid] += w
    for i in range(len(x)-1):
        c = x[i]+ " " +x[i+1]
        if  c in ngrams:
            vid = ngrams[c]["vid"]
            qVec[vid] += 1
    # print(qVec)
    # print(idf)
    qVec = normalize_tf(qVec, idf, qLen, avgLen)
    if rocchio:
        qVec = ro.rocchio(qVec, num, doc_vecs, docID2idx, queryDir+"/ans_train.csv")
    # print(qVec)
    
    qVec = csr_matrix(qVec)
    ans = []
    x = doc_vecs.dot(qVec.transpose())
    x = x.transpose()
    x = x / (  norm(qVec) * norm(doc_vecs, axis=1) )
    x = np.asarray(x).reshape(-1)

    for i, score in enumerate( tqdm(x, desc="write ans") ):
        ans.append([ score, docIDs[i] ]  )
        

    ans.sort(key=lambda x: x[0], reverse=True)

    out.write(num+",")
    for i in range(100):
        out.write( ans[i][1] + " ")
        # print( str(ans[i][0]) +" "+ ans[i][1])
    out.write("\n")


