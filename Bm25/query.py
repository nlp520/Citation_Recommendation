from Bm25.invdx import build_data_structures
from Bm25.rank import score_BM25
import operator


class QueryProcessor:
	def __init__(self, queries, corpus):
		self.queries = queries
		self.index, self.dlt = build_data_structures(corpus)

	def run(self):
		result = self.run_query(self.queries)
		return result

	def run_query(self, query):
		query_result = dict()
		#每个单词
		for term in query:
			if term in self.index:
				doc_dict = self.index[term] # retrieve index entry
				for docid, freq in doc_dict.items(): #for each document and its word frequency
					# print(len(doc_dict),' ',term, " ", docid, " ", freq, " ",self.dlt.get_length(docid))
					score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(self.dlt),
									   dl=self.dlt.get_length(docid), avdl=self.dlt.get_average_length()) # calculate score
					if docid in query_result: #this document has already been scored once
						query_result[docid] += score
					else:
						query_result[docid] = score
		return query_result