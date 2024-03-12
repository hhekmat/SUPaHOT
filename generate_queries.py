import os
import random
from openai import OpenAI

client = OpenAI(
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
        for i in range(50):
            curr_subset = random.choices(resource_labels, k=20)
            prompt_subset = random.choices(curr_subset, k=random.randint(1, 3))
            if i % 10 < 8:
                output_path = os.path.join("./queries/train", patient[:-13] + str(i) + '.txt')
            elif i % 10 < 9:
                output_path = os.path.join("./queries/validation", patient[:-13] + str(i) + '.txt')
            else:
                output_path = os.path.join("./queries/test", patient[:-13] + str(i) + '.txt')
            prompt = "Here is a subset of data points from a patient's medical record:\n" + "\n".join(prompt_subset) + "\n\nPretend you are a patient curious about an aspect of your medical history. Come up with a query that this patient might have regarding their medical data. At least one or more medical data points from the provided list should be sufficient to answer the query. Make the question realistic, simple, and non-technical. For example, 'What are my current medicines?' or 'When was my last shot?' or 'What were the complications of my last heart procedure?'."
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
                    for resource in curr_subset:
                        file.write(resource + '\n')

if __name__ == "__main__":
    generate_queries()