import together
import json
import os

SYSTEM_PROMPT = "You are a helpful medical assistant, users ask you questions pertaining to their health care information. You will help and be as concise and clear as possible."
task_1_prompt = "Given a query and a resource from a patient's medical record, your job is to determine if the resource could potentially be relevant to providing an answer to the patient's query about their medical history. Respond only with 'True' if the resource may be relevant, or 'False' if the resource would not be helpful at all in providing the patient an answer to their question." 
task_2_prompt = "Given an excerpt of a JSON object corresponding to a resource from a patient's FHIR medical records, your job is to provide a brief (1 to 2 sentence) natural language summary of the JSON. Don't explicitly mention that it's a JSON."
task_3_prompt = "You will be given a query from a patient who is inquiring about their medical records and a list of summaries (seperated by commas) of medical resources from the patient's medical record. Answer the patient's query using relevant information from the summaries."

system_prompt_task_1 = SYSTEM_PROMPT + " " + task_1_prompt
system_prompt_task_2 = SYSTEM_PROMPT + " " + task_2_prompt
system_prompt_task_3 = SYSTEM_PROMPT + " " + task_3_prompt

test_line_1 = {"query": "\"When was the last time my hemoglobin levels in my blood were checked?\"", "resource": "Observation Hemoglobin [Mass/volume] in Blood 02-28-2021", "label": "True"}
test_line_2 = {"resource_label": "\"{'fullUrl': 'urn:uuid:b10183cf-fe17-28a2-8786-533410e84e13', 'resource': {'resourceType': 'Procedure', 'id': 'b10183cf-fe17-28a2-8786-533410e84e13', 'meta': {'profile': ['http://hl7.org/fhir/us/core/StructureDefinition/us-core-procedure']}, 'status': 'completed', 'code': {'coding': [{'system': 'http://snomed.info/sct', 'code': '171207006', 'display': 'Depression screening (procedure)'}], 'text': 'Depression screening (procedure)'}, 'subject': {'reference': 'urn:uuid:9c3df38a-d3b7-2198-3898-51f9153d023d'}, 'encounter': {'reference': 'urn:uuid:c20c529f-99b2-3c7e-fec2-ada526b7fd37'}, 'performedPeriod': {'start': '2020-07-07T01:13:45-04:00', 'end': '2020-07-07T01:25:41-04:00'}, 'location': {'reference': 'Location?identifier=https://github.com/synthetichealth/synthea|586f94bd-51f7-39f5-92eb-f5fa92657469', 'display': 'NEWTON-WELLESLEY UROLOGY, PC'}}, 'request': {'method': 'POST', 'url': 'Procedure'}}\"", "summary": "This record documents a completed depression screening procedure that took place on July 7, 2020, at NEWTON-WELLESLEY UROLOGY, PC."}
test_line_3 = {"query": "\"What were the results of my depression screening on 07-07-2020?\"", "resource_summaries": ["This record documents a completed depression screening procedure that took place on July 7, 2020, at NEWTON-WELLESLEY UROLOGY, PC.\n"], "answer": "Based on the summary provided, the results of your depression screening on July 7, 2020, at NEWTON-WELLESLEY UROLOGY, PC, were documented in your medical record. To obtain specific details about the results, you may need to request a copy of your medical records from that facility."}




def get_instruct_template(user_msg, model_answer, system_prompt):
    instruct_template = \
    f"""<s>[INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {user_msg} [/INST] {model_answer} </s>"""
    return instruct_template#.format(user_msg=user_msg, model_answer=model_answer, system_prompt=system_prompt)

def write_task_1():
    types = ["train", "val"]
    for type in types:
        read_path = f"ft_datasets/task_1/task_1_{type}.jsonl"
        write_path = f"llama_ft_datasets/task_1/task_1_{type}.jsonl"
        queries = []
        resources = []
        labels = []
        with open(read_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                queries.append(data['query'])
                resources.append(data['resource'])
                labels.append(data['label'])
        
        with open(write_path, 'w') as file:
            for i in range(len(queries)):
                user_msg = f"{queries[i]} {resources[i]}"
                model_answer = labels[i]
                system_prompt = system_prompt_task_1
                
                line = {"text": get_instruct_template(user_msg, model_answer, system_prompt)}
                file.write(json.dumps(line) + '\n')

def write_task_2():
    types = ["train", "val"]
    for type in types:
        read_path = f"ft_datasets/task_2/task_2_{type}.jsonl"
        write_path = f"llama_ft_datasets/task_2/task_2_{type}.jsonl"
        resources = []
        summaries = []
        with open(read_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if 'resource_label' in data:
                    resources.append(data['resource_label'])
                else:
                    resources.append(data['resource'])
                summaries.append(data['summary'])
        
        with open(write_path, 'w') as file:
            for i in range(len(resources)):
                #user_msg = f"{resources[i]}"
                user_msg = resources[i]
                model_answer = summaries[i]
                system_prompt = system_prompt_task_2
                line = {"text": get_instruct_template(user_msg, model_answer, system_prompt)}
                file.write(json.dumps(line) + '\n')

def write_task_3():
    types = ["train", "val"]
    for type in types:
        read_path = f"ft_datasets/task_3/task_3_{type}.jsonl"
        write_path = f"llama_ft_datasets/task_3/task_3_{type}.jsonl"
        queries = []
        resource_summaries = []
        answers = []
        with open(read_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                queries.append(data['query'])
                resource_summaries.append(data['resource_summaries'])
                answers.append(data['answer'])
        
        with open(write_path, 'w') as file:
            for i in range(len(queries)):
                user_msg = queries[i] + ' '
                user_msg += ", ".join(resource_summaries[i])
                model_answer = answers[i]
                system_prompt = system_prompt_task_3
                
                line = {"text": get_instruct_template(user_msg, model_answer, system_prompt)}
                file.write(json.dumps(line) + '\n')

'''def remove_newlines():
    for root, dirs, files in os.walk('./llama_ft_datasets'):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            modified_lines = []
            for line in lines:
                data = json.loads(line)
                data['text'] = ' '.join(data['text'].split())
                modified_lines.append(json.dumps(data) + '\n')
            with open(file_path, 'w') as f:
                f.writelines(modified_lines)'''


def main():
    pass

if __name__ == '__main__':
    main()