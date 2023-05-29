# Код для многопоточного создания spacy-документов

# %%
import spacy
from spacy.tokens import DocBin
en_nlp_lg = spacy.load("en_core_web_lg")

# %%
filename_translated_all = "translated/xaa"
with open(filename_translated_all, "rt", encoding="utf-8") as f:
    translated_all = [line.rstrip() for line in f.readlines()]
len(translated_all)

# %%
docs_bin = DocBin()
for doc in en_nlp_lg.pipe(translated_all, n_process=-1):
    docs_bin.add(doc)

bytes_data = docs_bin.to_bytes()

with open("spaced/xaa", "wb") as file:
    file.write(bytes_data)
