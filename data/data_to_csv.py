import os
import pandas as pd

data_path = "data/CEFR"
data = []
for level in os.listdir(data_path)[1:]:
    level_dir = os.path.join(data_path, level)
    for text_file in os.listdir(level_dir):
        try:
            with open(os.path.join(level_dir, text_file), "r") as f:
                text = f.read()
            data.append({"text": text, "label": level})
        except UnicodeDecodeError:
            print(level_dir, text_file)

print(len(data))

df = pd.DataFrame(data)
df.to_csv(os.path.join(data_path, "cefr_leveled_texts.csv"), index=False)
