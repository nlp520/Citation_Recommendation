from math import log

k1 = 1.2
k2 = 100
b = 0.75 #对于文档长度的缩放程度
R = 0.0


def score_BM25(n, f, qf, r, N, dl, avdl):
	'''
	n=len(doc_dict), f=freq, qf=1, r=0, N=len(self.dlt),dl=self.dlt.get_length(docid), avdl=self.dlt.get_average_length()
	:param n: 包含这个单词的文档总数
	:param f:单词出现的频率，
	:param qf:
	:param r:
	:param N:查询文档的总数，
	:param dl: 文档的长度
	:param avdl: 平均文档的长度
	:return:
	'''
	K = compute_K(dl, avdl)
	first = log( ( (r + 0.5) / (R - r + 0.5) ) / ( (n - r + 0.5) / (N - n - R + r + 0.5)) )
	second = ((k1 + 1) * f) / (K + f)
	third = ((k2+1) * qf) / (k2 + qf)
	return first * second * third


def compute_K(dl, avdl):
	return k1 * ((1-b) + b * (float(dl)/float(avdl)) )