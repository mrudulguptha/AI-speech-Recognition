import os

ALIGN_PATH = "align"
OUTPUT_FILE = "labels.txt"

labels = []

for speaker in os.listdir(ALIGN_PATH):
    speaker_path = os.path.join(ALIGN_PATH, speaker)

    if not os.path.isdir(speaker_path):
        continue

    for file in os.listdir(speaker_path):
        if file.endswith(".align"):
            file_path = os.path.join(speaker_path, file)

            words = []
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) == 3:
                        word = parts[2]

                        # ignore silence
                        if word == "sil":
                            continue

                        words.append(word)

            name = file.replace(".align", "")
            labels.append(f"{speaker}_{name} {' '.join(words)}")

with open(OUTPUT_FILE, "w") as f:
    for line in labels:
        f.write(line + "\n")

print("Labels created!")