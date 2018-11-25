from Bm25.query import QueryProcessor
import operator
from nltk import word_tokenize

from Retrieval.train import cal_MAP
from process import pickleload, jsonsave
from collections import OrderedDict
from tqdm import tqdm

from rougetest.test_bleu import test_bleu
from rougetest.test_score import test_score
from random import randint

from similar import process, getTopVsmScore

def Bm25_similar():
	'''
	[
            {
             "citStr":"" 引用的作者和年份,
             "context":"", 整个引用片段
             "up_source_tokens":"",
             "down_source"_tokens:"",
             "target_tokens":""
             "citations":[
                          {
                          	"up_source_tokens":
                          	"down_source_tokens":
                          	"target_tokens":
                          }
                           ...
                          ]
            }
            ......

        ]
	:return:
	'''
	datas = pickleload("../data2/random_train_data.pkl", "../data2/random_train_data.pkl")
	datas = datas[len(datas)*4//5:len(datas)]
	MAPS = 0
	precisions= 0
	recalls = 0
	for data in tqdm(datas):
		target_up_content = process(data["up_source_tokens"]).split(" ")
		target_down_content = process(data["down_source_tokens"]).split(" ")
		target_content = process(data["target_tokens"])
		content_tokens = target_up_content #+ target_down_content
		citations = data["citations_tokens"]
		citation_content_dict = dict()
		citation_target_dict = dict()
		index = 0
		# print(len(citations))
		ref_lis = []
		for citation in citations:
			sel_up_content = process(citation["up_source_tokens"]).split(" ")
			sel_down_content =process( citation["down_source_tokens"]).split(" ")
			sel_target = process(citation["target_tokens"])
			citation_content_tokens = sel_up_content + sel_target.split(" ") +sel_down_content
			citation_content_dict[str(index)] = citation_content_tokens
			citation_target_dict[str(index)] = sel_target
			if citation['label'] == 1:
				ref_lis.append(index)
			index += 1

		pre_lis = getBm25TopSimilar(content_tokens, citation_content_dict, num=5)

		precision, recall ,MAP = cal_MAP(ref_lis, pre_lis)
		MAPS += MAP
		precisions += precision
		recalls += recall

	MAPS /= len(datas)
	precisions /= len(datas)
	recalls /= len(datas)
	print("MAP：%.4f  P：%.4f  R：%.4f" % (MAPS, precisions, recalls))


def getBm25TopSimilar(qp, cp, num=3):
	'''
	返回按照相似度由高到低的排名
	:param qp: 要检索的文档 list[]
	:param cp: 候选的数据集 dict{index : [list]}
	:return:
	'''
	queries = qp
	corpus = cp
	# print("queries:",queries)
	# print("corpus:",corpus)
	proc = QueryProcessor(queries, corpus)
	result = proc.run()
	sorted_x = sorted(result.items(), key=operator.itemgetter(1))
	sorted_x.reverse()
	# index = 0
	# for i in sorted_x[:100]:
	# 	tmp = (qid, i[0], index, i[1])
	# 	print('{:>1}\tQ0\t{:>4}\t{:>2}\t{:>12}\tNH-BM25'.format(*tmp))
	# 	index += 1
	# print(sorted_x[0][0], sorted_x[0][1])
	pre_lis = []
	for i in range(num):
		pre_lis.append(int(sorted_x[i][0]))
	return pre_lis

def random():
	import random
	idf_dic = pickleload("../data2/idf.pkl", "idf.pkl")
	random.seed = 1
	datas = pickleload("../data2/train_data3.pkl", "../data/train_data.pkl")
	datas = datas[len(datas)*4//5 : len(datas)]
	result_lis = []
	all_count = 0
	true_count = 0
	for data in tqdm(datas):
		target_content = data["target_tokens"]
		citations = data["citations_tokens"]
		citation_target_dict = dict()
		index = 0
		# print(len(citations))
		new_citations = []
		for citation in citations:
			sel_target = citation["target_tokens"]
			citation_target_dict[index] = sel_target
			index += 1
			new_citations.append(sel_target)
		# print(len(citation_content_dict))
		random_index = randint(0, len(citation_target_dict)-1)
		ref = getTopVsmScore(idf_dic, target_content, new_citations)
		all_count+= 1
		if random_index == ref:
			true_count += 1

	print(true_count/all_count)

if __name__ == '__main__':
	Bm25_similar()
	# random()
	pass



