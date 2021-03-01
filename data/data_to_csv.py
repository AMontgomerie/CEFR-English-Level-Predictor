import os
import pandas as pd


if __name__ == "__main__":
    data = []
    for level in os.listdir("data")[1:]:
        level_dir = os.path.join("data", level)
        for text_file in os.listdir(level_dir):
            try:
                with open(os.path.join(level_dir, text_file), "r") as f:
                    text = f.read()
                data.append({"text": text, "label": level})
            except UnicodeDecodeError:
                print(level_dir, text_file)

    print(len(data))

    df = pd.DataFrame(data)
    df.to_csv(os.path.join("data", "cefr_leveled_texts.csv"), index=False)
