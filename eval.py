import os
import sys
import numpy as np
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

def load_data(task, m):
    m_dict = read_txt_files_into_dict(f"./task_{task}/output/meditron/test")
    if m == 0:
        print('reading into meditron')
        m_dict = read_txt_files_into_dict(f"./task_{task}/output/meditron/test")
    elif m == 1:
        print('reading into llama')
        m_dict = read_txt_files_into_dict(f"./task_{task}/output/llama/test")
    elif m == 2:
        m_dict = read_txt_files_into_dict(f"./task_{task}/output/meditron_ft/test")
    oracle_dict = read_txt_files_into_dict(f"./task_{task}/output/oracle/test")
    
    m_list = []
    oracle_list = []

    overlap = list(set(m_dict.keys()).intersection(set(oracle_dict.keys())))
    for key in overlap:
        m_list.append(m_dict[key])
        oracle_list.append(oracle_dict[key])
    
    return oracle_list, m_list

def tokenize(string_list):
        tokenized_list = []
        for string in string_list:
            # Tokenize the string into words
            words = word_tokenize(string)
            tokenized_list.append(words)
        return tokenized_list

def resource_label_overlap(refs, hyps):
    refs = [ref.split('\n') for ref in refs]
    hyps = [hyp.split('\n') for hyp in hyps]
    num = np.array([len(set(refs[i]).intersection(set(hyps[i]))) for i in range(len(refs))])
    denomP = np.array([len(hyps[i]) for i in range(len(hyps))])
    denomR = np.array([len(refs[i]) for i in range(len(refs))])
    P = num / denomP
    R = num / denomR
    F1 = (2 * P * R) / (P + R)
    return np.mean(P), np.mean(R), np.mean(F1)

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
    if len(sys.argv) > 2:
        task = int(sys.argv[1])
        m = int(sys.argv[2])
        model = 'placeholder'
        if m == 0:
            model = 'meditron'
        elif m == 1:
            model = 'llama'
        elif m == 2:
            model = 'llama_ft'
        refs, hyps = load_data(task, m)

        if task == 1:
            p, r, f1 = resource_label_overlap(refs, hyps)
            report += 'precision: ' + str(p) + ' recall: ' + str(r) + ' f1: ' + str(f1) + '\n'
        elif task == 2 or task == 3: 
            t_refs = tokenize(refs)
            t_hyps = tokenize(hyps)
            b, avg_s_b = bleu(t_refs, t_hyps)
            report += 'corpus bleu score: ' + str(b) + '\n'
            report += ' ' + 'avg sentence bleu score: ' + str(avg_s_b) + '\n'

            r = rouge(t_refs, t_hyps)
            report += 'rouge score: ' + str(r) + '\n'

            bP, bR, bF1 = bertscore(refs, hyps)
            print('bertSCORE precision: ' + str(bP))
            print('bertSCORE recall: ' + str(bR))
            print('bertSCORE F1: ' + str(bF1))
            report += ' bertSCORE precision: ' + str(bP) + ' bertSCORE recall: ' + str(bR) + 'bertSCORE F1: ' + str(bF1)

        else:
            print("Invalid task number. Please choose 1, 2, or 3.")
        with open(f'./task_{task}_{model}_result.txt', 'w') as file:
            file.write(report)
    else:
        print("Please provide a task number and finetune bool (0 or 1) as a command-line argument.")


