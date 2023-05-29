# %%
from multiprocessing import Pool

# %%
def foo(zip_ru_en):
    from simalign import SentenceAligner
    myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")
    all_alignments = []
    for ru_tokens, en_tokens in zip_ru_en:
        all_alignments.append(myaligner.get_word_aligns(ru_tokens, en_tokens))
    return all_alignments

# %%

# rename file to main before running
if __name__ == '__main__':
    with open("tokens/xaa", "rt", encoding='utf-8') as file:
        translated_tokens = [line.split() for line in file]
    with open("subtitles_ru_en_chunks/ru/xaa", "rt", encoding='utf-8') as file:
        ru_tokens = [line.split() for line in file]

    folder_name = "alignments"
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
    arguments = []
    for i in range(latest_chunk, 500000, chunk_size):
        tmp = []
        for pair in zip(ru_tokens[i:i+chunk_size], translated_tokens[i:i+chunk_size]):
            tmp.append(pair)
        arguments.append(tmp)
    
    with Pool(25) as p:
        aligned_chunks = p.map(foo, arguments)

    for idx, chunk in enumerate(aligned_chunks):
        for i in range(idx*chunk_size, (idx+1)*chunk_size):
            mwmf[i] = " ".join([f"{x}-{y}" for x, y in chunk[i%chunk_size]["mwmf"]])
            itermax[i] = " ".join([f"{x}-{y}" for x, y in chunk[i%chunk_size]["itermax"]])
            inter[i] = " ".join([f"{x}-{y}" for x, y in chunk[i%chunk_size]["inter"]])

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