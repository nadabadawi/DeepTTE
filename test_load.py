import json

def load_json_file(file_path):
    list_of_dicts = []

    with open(file_path, 'r') as file:
        for line in file:
            try:
                json_dict = json.loads(line)
                list_of_dicts.append(json_dict)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line: {line}. Error: {e}")
    
    length = [len(x['lngs']) for x in list_of_dicts]

    return (length, list_of_dicts)

# Example usage:
file_path = './data/train_00'
(length, list_of_dicts) = load_json_file(file_path)
print(list_of_dicts)
print("Length ", len(length))