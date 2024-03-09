import os
from nltk.tokenize import word_tokenize
from nltk.translate import bleu_score
from rouge_metric import PyRouge
import bert_score

def read_txt_files_into_dict(directory):
    # Dictionary to hold contents of txt files from each directory
    content_dict = {}
    for filename in os.listdir(directory):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            # Construct the full file path
            filepath = os.path.join(directory, filename)
            # Read the content of the file
            with open(filepath, 'r') as file:
                content = file.read()
                # Use filename without extension as key
                key = os.path.splitext(filename)[0]
                content_dict[key] = content
    return content_dict

def load_data():
    meditron_dict = read_txt_files_into_dict("./generated_outputs/test")
    oracle_dict = read_txt_files_into_dict("./oracle/test")
    
    meditron_list = []
    oracle_list = []

    overlap = list(set(meditron_dict.keys()).intersection(set(oracle_dict.keys())))
    for key in overlap:
        meditron_list.append(meditron_dict[key])
        oracle_list.append(oracle_dict[key])
    
    return oracle_list, meditron_list

def tokenize(string_list):
        tokenized_list = []
        for string in string_list:
            # Tokenize the string into words
            words = word_tokenize(string)
            tokenized_list.append(words)
        return tokenized_list

def bleu(refs, hyps):
    smoother = bleu_score.SmoothingFunction()
    def get_avg_sentence_bleu(refs, hyps):
        sum = 0
        for i in range(len(refs)):
            curr = bleu_score.sentence_bleu(refs[i], hyps[i], smoothing_function=smoother.method3)
            sum += curr
        return sum / len(refs)
    c_bleu = bleu_score.corpus_bleu(refs, hyps, smoothing_function=smoother.method3)
    avg_s_bleu = get_avg_sentence_bleu(refs, hyps)
    return c_bleu, avg_s_bleu

def rouge(refs, hyps):
    rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
    scores = rouge.evaluate_tokenized(hyps, refs)
    return scores

def bertscore(refs, hyps):
    P, R, F1 = bert_score.score(hyps, refs, lang='en', verbose=True, model_type='microsoft/deberta-xlarge-mnli')
    avg_P = sum(P) / len(P)
    avg_R = sum(R) / len(R)
    avg_F1 = sum(F1) / len(R)
    return avg_P, avg_R, avg_F1

if __name__ == "__main__":

    report = ''

    oracle_list, meditron_list = load_data()
    t_oracle_list = tokenize(oracle_list)
    t_meditron_list = tokenize(meditron_list)

    b, avg_s_b = bleu(t_oracle_list, t_meditron_list)
    report += 'corpus bleu score: ' + str(b) + '\n'
    report += ' ' + 'avg sentence bleu score: ' + str(avg_s_b) + '\n'

    r = rouge(t_oracle_list, t_meditron_list)
    report += 'rouge score: ' + str(r) + '\n'

    bP, bR, bF1 = bertscore(oracle_list, meditron_list)
    print('bertSCORE precision: ' + str(bP))
    print('bertSCORE recall: ' + str(bR))
    print('bertSCORE F1: ' + str(bF1))
    report += ' bertSCORE precision: ' + str(bP) + ' bertSCORE recall: ' + str(bR) + 'bertSCORE F1: ' + str(bF1)

    with open('/mnt/data/report_eval.txt', 'a') as file:
        file.write(report)
