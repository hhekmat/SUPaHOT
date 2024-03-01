import os
from openai import OpenAI
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def generate_queries():
    resource_types = ['allergyIntolerance', 'Condition', 'Encounter', 'Immunization', 'MedicationRequest', 'Observation', 'Procedure']
    resources_data_folder = "./all_resources"
    resources_data_folder = "./all_resources"
    file_names = os.listdir(resources_data_folder)
    for patient in file_names:
        previous_queries = []
        resources_file = os.path.join(resources_data_folder, patient)
        for rt in resource_types:
            fhir_data_of_rt = []
            with open(resources_file, 'r') as file:
                for line in file:
                    if line.startswith(rt):
                        fhir_data_of_rt.append(line.strip())
            for i in range(10):
                if i == 0:
                    output_path = os.path.join("./queries/test", patient[:-13] + rt + str(i) + '.txt')
                    output_path = os.path.join("./queries/test", patient[:-13] + rt + str(i) + '.txt')
                elif i == 1:
                    output_path = os.path.join("./queries/validation", patient[:-13] + rt + str(i) + '.txt')
                    output_path = os.path.join("./queries/validation", patient[:-13] + rt + str(i) + '.txt')
                else:
                    output_path = os.path.join("./queries/train", patient[:-13] + rt + str(i) + '.txt')
                prompt = "Previous Queries:\n" + "\n".join(previous_queries) + "\n\nFHIR Data Resources:\n" + "\n".join(fhir_data_of_rt) + "\n\nGenerate a succinct query:"
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="gpt-3.5-turbo", 
                    )               
                if chat_completion.choices and chat_completion.choices[0].message:
                    generated_query = chat_completion.choices[0].message.content
                    with open(output_path, 'w') as file:
                        file.write(generated_query + '\n')
                        for resource in fhir_data_of_rt:
                            file.write(resource + '\n')
                else:
                    with open(output_path, 'w') as file:
                        file.write('nothing was returned (or ur calling chat completion wrong lmao)')
                        for resource in fhir_data_of_rt:
                            file.write(resource + '\n')
                previous_queries.append(generated_query)

if __name__ == "__main__":
    generate_queries()