import yaml
import json
import os

# Config load
with open('config.yml', 'r', encoding='utf-8') as config_file:
    config = yaml.safe_load(config_file)



## Language settings ##
valid_language_codes = []
lang_directory = "lang"
current_language_code = config['LANGUAGE']

for filename in os.listdir(lang_directory):
    if filename.startswith("lang.") and filename.endswith(".json") and os.path.isfile(
            os.path.join(lang_directory, filename)):
        language_code = filename.split(".")[1]
        valid_language_codes.append(language_code)


def load_current_language():
    lang_file_path = os.path.join(
        lang_directory, f"lang.{current_language_code}.json")
    with open(lang_file_path, encoding="utf-8") as lang_file:
        current_language = json.load(lang_file)
    return current_language

# personas loader for personas
def load_personas():
    persona_list = {}
    for file_name in os.listdir("personas"):
        if file_name.endswith('.txt'):
            file_path = os.path.join("personas", file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                # Use the file name without extension as the variable name
                persona = file_name.split('.')[0]
                persona_list[persona] = file_content
    return persona_list