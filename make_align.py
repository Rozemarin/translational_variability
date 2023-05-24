from tqdm import tqdm
from spacy.tokens import DocBin
import spacy

from simalign import SentenceAligner
myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

with open("corpora/subtitles/ru_subtitles_spacy_dump.bin", "rb") as f:
    restored_bytes_data = f.read()

nlp = spacy.blank("ru")
doc_bin = DocBin().from_bytes(restored_bytes_data)
ru_all_docs = list(doc_bin.get_docs(nlp.vocab))

with open("corpora/subtitles/translations/opus10_spacy.bin", "rb") as file:
    en_translated_bytes_data = file.read()

nlp = spacy.blank("en")
doc_bin = DocBin().from_bytes(en_translated_bytes_data)
en_translated_docs = list(doc_bin.get_docs(nlp.vocab))

folder_name = "corpora/subtitles/translations"
fname_mwmf = f"{folder_name}/mwmf"
fname_itermax = f"{folder_name}/itermax"
fname_inter = f"{folder_name}/inter"

with open(fname_mwmf, "rt", encoding="utf-8") as f:
    mwmf = [line.rstrip() for line in f.readlines()]
with open(fname_itermax, "rt", encoding="utf-8") as f:
    itermax = [line.rstrip() for line in f.readlines()]
with open(fname_inter, "rt", encoding="utf-8") as f:
    inter = [line.rstrip() for line in f.readlines()]

chunk_size = 10000
latest_chunk = 0
for chunk_start in range(latest_chunk, 100000, chunk_size):

    for i in tqdm(range(chunk_start, chunk_start + chunk_size)):
        if mwmf[i] != "":
            continue
        if len(en_translated_docs) == 0:
            continue
        ru_tokens = [token.text for token in ru_all_docs[i]]
        translated_tokens = [token.text for token in en_translated_docs[i]]
        # print(f"{ru_tokens}\n{translated_tokens}\n")
        src, trg = (ru_tokens, translated_tokens)
        alignments = myaligner.get_word_aligns(src, trg)
        mwmf[i] = " ".join([f"{x}-{y}" for x, y in alignments["mwmf"]])
        itermax[i] = " ".join([f"{x}-{y}" for x, y in alignments["itermax"]])
        inter[i] = " ".join([f"{x}-{y}" for x, y in alignments["inter"]])

    with open(fname_mwmf, 'w', encoding="utf-8") as f:
        for line in mwmf:
            f.write(line)
            f.write('\n')

    with open(fname_itermax, 'w', encoding="utf-8") as f:
        for line in itermax:
            f.write(line)
            f.write('\n')

    with open(fname_inter, 'w', encoding="utf-8") as f:
        for line in inter:
            f.write(line)
            f.write('\n')