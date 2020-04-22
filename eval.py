import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='vsm')
parser.add_argument('-i', type=str, dest="my_ans")
args = parser.parse_args()


topN = 30

ansfile = "queries/ans_train.csv"
resfile = args.my_ans #"my_ans_train.csv"


ans_list = []
res_list = []

ans = open(ansfile)
next(ans)


for line in ans:
    # print(line)
    line = re.split(",| |\n", line.strip())
    # print(line)
    ans_list.append(set(line[1:]))


res = open(resfile)
next(res)


for line in res:
    line = re.split(",| |\n", line.strip())
    res_list.append(line[1:topN+1])

# print(res_list)
# print(ans_list)
map = 0
for i, doc_list in enumerate(res_list):
    cnt = 0
    ap = 0
    for n, doc in enumerate(doc_list, start=1):
        if doc in ans_list[i]:
            cnt += 1
        ap += cnt / n
    ap /= topN
    print(ap)
    map += ap

print("map: ", map / len(ans_list))
