import json
import csv

with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

def extract_texts(obj, key_path=("commentary", "text", "text")):
    texts = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == key_path[0]:
                texts.extend(extract_texts(value, key_path=key_path[1:]))
            else:
                texts.extend(extract_texts(value, key_path=key_path))
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_texts(item, key_path=key_path))
    elif not key_path:
        texts.append(obj)
    return texts

texts = extract_texts(data)

output_file = 'extracted_posts.csv'
with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Post"])  # Add header
    for text in texts:
        writer.writerow([text])  # Write each text as a new row

print(f"Extracted texts have been saved to {output_file}")