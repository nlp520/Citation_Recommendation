from rougetest.bleu import Bleu


N_SIZE = 2


def test_count_bp():
    cand = '我是中国人'
    ref = '重视啊啊啊啊我啊啊我了'
    bleu = Bleu(N_SIZE)
    bp = bleu.count_bp(cand, ref)
    print('BP: {}'.format(bp))


def test_count_bp2():
    cand = '我是中国人当外人大气'
    ref = '重视啊啊'
    bleu = Bleu(N_SIZE)
    bp = bleu.count_bp(cand, ref)
    print('BP: {}'.format(bp))


def test_add_inst():
    cand = '13'
    ref = '13'
    bleu = Bleu(N_SIZE)
    bleu.add_inst(cand, ref)
    match_ngram = bleu.match_ngram
    candi_ngram = bleu.candi_ngram
    print('match_ngram: {}'.format(match_ngram))
    print('candi_ngram: {}'.format(candi_ngram))


def test_bleu(cand, ref, N_SIZE=1):
    bleu = Bleu(N_SIZE)
    bleu.add_inst(cand, ref)
    # try:
    s = bleu.get_score()
    # except:
    #     print(bleu.match_ngram)
    #     print(bleu.candi_ngram)
    #     print("cand",cand)
    #     print("ref",ref)
    return s

def getBleu(cand, ref,N_SIZE=1):
    bleu = Bleu(N_SIZE)
    bleu.add_inst(cand, ref)
    s = bleu.get_score()
    return s

if __name__ == '__main__':
    # test_count_bp()
    # test_count_bp2()
    # test_ngram()
    test_bleu()
    # test_add_inst()
