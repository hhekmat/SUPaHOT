import os
# from nltk.translate import bleu_score
# from rouge_metric import PyRouge
# from bert_score import score

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
    else:
        print(f"Directory {directory} does not exist.")
    
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

def bleu(refs, hyps):
    '''
    list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    hypotheses = [hyp1, hyp2]
    '''
    bleu = bleu_score.corpus_bleu(refs, hyps)

def rouge(refs, hyps):
    '''
    hypotheses = [
    'how are you\ni am fine',  # document 1: hypothesis
    'it is fine today\nwe won the football game',  # document 2: hypothesis
    ]
    references = [[
        'how do you do\nfine thanks',  # document 1: reference 1
        'how old are you\ni am three',  # document 1: reference 2
    ], [
        'it is sunny today\nlet us go for a walk',  # document 2: reference 1
        'it is a terrible day\nwe lost the game',  # document 2: reference 2
    ]]
    '''
    rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
    scores = rouge.evaluate(hyps, refs)

def bertscore(refs, hyps):
    # 
    P, R, F1 = score(hyps, refs, lang='en', verbose=True)

if __name__ == "__main__":
    oracle_list, meditron_list = load_data()