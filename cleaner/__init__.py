from cleaner.hack_tricks_cleaner import redundant_strings as hack_tricks_redundant_strings
from cleaner.hack_tricks_cleaner import cutting_strings as hack_tricks_cutting_strings
from cleaner.metasploit_clearner import redundant_strings as metasploit_redundant_strings
from cleaner.metasploit_clearner import cutting_strings as metasploit_tricks_cutting_strings
from cleaner.mitre_cleaner import redundant_strings as mitre_redundant_strings
from cleaner.mitre_cleaner import cutting_strings as mitre_tricks_cutting_strings
from cleaner.payload_all_things_cleaner import redundant_strings as payload_all_things_redundant_strings
from cleaner.payload_all_things_cleaner import cutting_strings as payload_all_things_tricks_cutting_strings
from cleaner.red_team_cleaner import redundant_strings as red_team_redundant_strings
from cleaner.red_team_cleaner import cutting_strings as red_team_cutting_strings
import os
from tqdm.auto import tqdm
def specific_cleaner(input_dir="data/raw", output_dir="data/clean", subtask="hack_tricks", redundant_strings=[], cutting_strings=[]):
    os.makedirs(output_dir, exist_ok=True)
    subtask_folder = os.path.join(input_dir, subtask)
    subtask_folder_cleaned = os.path.join(output_dir, subtask)
    os.makedirs(subtask_folder_cleaned, exist_ok=True)
    for file in tqdm(os.listdir(subtask_folder)):
        file_path = os.path.join(subtask_folder, file)
        with open(file_path, "r") as f:
            text = f.read()
        if len(text) == 0:
            continue
        bad_chars = ["\u200b", "\x00"]
        for bad_char in bad_chars:
            text = text.replace(bad_char, "")
        for redundant_string in redundant_strings:
            for bad_char in bad_chars:
                redundant_string = redundant_string.replace(bad_char, "")
            text = text.replace(redundant_string, "")
        for cutting_string in cutting_strings:
            if cutting_string in text:
                text = cutting_string.join(text.split(cutting_string)[:-1])
        http_texts = []
        for text_split in text.split():
            if text_split.startswith("http"):
                http_texts.append(text_split)
        for http_text in http_texts:
            text = text.replace(http_text, "URL")
        for redundant_string in redundant_strings:
            for bad_char in bad_chars:
                redundant_string = redundant_string.replace(bad_char, "")
            text = text.replace(redundant_string, "")
        cleaned_file_path = os.path.join(subtask_folder_cleaned, file)
        with open(cleaned_file_path, "w") as f:
            f.write(text)
if __name__ == '__main__':
    skip_index = 0
    end_index = 100
    parsers = [(hack_tricks_redundant_strings, hack_tricks_cutting_strings), (metasploit_redundant_strings, metasploit_tricks_cutting_strings), (mitre_redundant_strings, mitre_tricks_cutting_strings), (payload_all_things_redundant_strings, payload_all_things_tricks_cutting_strings), (red_team_redundant_strings, red_team_cutting_strings)]
    db_names = ["hack_tricks", "metasploit", "mitre", "payload_all_things", "red_team"]
    for (parser, db_name) in zip(parsers[skip_index:end_index], db_names[skip_index:end_index]):
        redundant_strings, cutting_strings = parser
        specific_cleaner(subtask=db_name, redundant_strings=redundant_strings, cutting_strings=cutting_strings)