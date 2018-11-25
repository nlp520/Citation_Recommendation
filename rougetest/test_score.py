import json

from process import jsonsave
from rougetest.bleu import Bleu
from rougetest.rouge import RougeL


def read_json(filename) -> list:
    """读取json文件

    Returns:
        list -- N * 2列的列表，每列分别保存参考答案和预测答案
    """
    with open(filename, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def test_score(filename, n_size=1):
    # init all argument
    data = read_json(filename)

    rouge_eval = RougeL()
    bleu_eval = Bleu(n_size)
    for idx, (ref_key, cand_key) in enumerate(data):
        ref_sent = data[idx][ref_key]
        cand_sent = data[idx][cand_key]

        # print("参考答案：",ref_sent)
        # print("预测答案：",cand_sent)
        # print("--------------------------------------------------")

        rouge_eval.add_inst(cand_sent, ref_sent)
        bleu_eval.add_inst(cand_sent, ref_sent)

    bleu_score = bleu_eval.get_score()
    rouge_score = rouge_eval.get_score()
    print('bleu score: {}, rouge score: {}'.format(bleu_score, rouge_score))
    return bleu_score

if __name__ == '__main__':
    # test_scores()
    # test_score8()
    # test_score5()

    test_score("./rougetest/data/test_data.json", n_size=2)







