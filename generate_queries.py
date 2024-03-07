import os
import random
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def generate_queries():
    resources_data_folder = "./all_resources"
    file_names = os.listdir(resources_data_folder)
    for patient in file_names:
        resources_file = os.path.join(resources_data_folder, patient)
        resource_labels = []
        with open(resources_file, 'r') as file:
            for line in file:
                resource_labels.append(line.strip())
        for i in range(100):
            curr_subset = random.choices(resource_labels, random.randint(1, 5))
            if i < 80:
                output_path = os.path.join("./queries/train", patient[:-13] + str(i) + '.txt')
            elif i < 90:
                output_path = os.path.join("./queries/validation", patient[:-13] + str(i) + '.txt')
            else:
                output_path = os.path.join("./queries/test", patient[:-13] + str(i) + '.txt')
            prompt = "Here is a subset of a patient's FHIR Data Resources:\n" + "\n".join(curr_subset) + "\n\nGenerate a query that a patient with this history would have. Make it succinct, specific, non-technical, and like a normal person. For example, 'What are my current medicines?' or 'When was my last shot?'" 
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

if __name__ == "__main__":
    generate_queries()