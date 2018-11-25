from rougetest.rouge import RougeL

rouge = RougeL()


def test_lcs():
    cand = '中华人民共和国中的人'
    ref = '中国人民共和国的'
    lcs = rouge.lcs(cand, ref)
    print('lcs: {}'.format(lcs))


def test_lcs2():
    cand = '中华人民'
    ref = '中国人民共和国的'
    lcs = rouge.lcs(cand, ref)
    print('lcs: {}'.format(lcs))


def test_rouge(cand, ref):
    rouge.add_inst(cand, ref)
    score = rouge.get_score()
    return score


if __name__ == '__main__':
    test_lcs()
    test_lcs2()
