import numpy as np
import re

def rocchio(qVec, qid, doc_vecs, docID2idx, ansfile, a = 1, b = 0.8, c = 0):

    # ansfile = "queries/ans_train.csv"

    ans_list = []
    ans = open(ansfile)
    next(ans)

    dim = len(qVec)
    tmp = np.zeros(dim)

    for line in ans:
        # print(line)
        line = re.split(",| |\n", line.strip())
        if line[0] == qid:
            
            ans_list = list( map( lambda x: docID2idx[x],  line[1:] ) )
            print("ans list:  ", ans_list)
            n = len(ans_list)
            for i in ans_list:
                tmp += doc_vecs[i]
            qVec = a * qVec + b / n * tmp
    return qVec
