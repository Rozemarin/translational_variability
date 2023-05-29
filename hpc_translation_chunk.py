# Код для эффективной работы с переводчиком при наличии GPU

# %%
from tqdm import tqdm

# %%
from easynmt import EasyNMT
model = EasyNMT('mbart50_m2m')

# %%
foldernale = 'subtitles_ru_en_chunks'
filename_src = 'xaa'
filename_translated_all = f"translated/{filename_src}"

# %%
with open(f"{foldernale}/ru/{filename_src}", "rt", encoding="utf-8") as f:
    ru_all = [line.strip() for line in f]

# %%
with open(filename_translated_all, "rt", encoding="utf-8") as f:
    translated_all = [line.rstrip() for line in f.readlines()]
len(translated_all)

# %%
chunk_size = 30000
latest_chunk = 93500

for chunk_start in tqdm(range(latest_chunk, 500000, chunk_size)):

    translations = model.translate(ru_all[chunk_start:chunk_start+chunk_size], source_lang='ru', target_lang='en', max_length=200)
    translated_all[chunk_start:chunk_start+chunk_size] = translations

    with open(filename_translated_all, 'w', encoding="utf-8") as f:
        for line in translated_all:
            f.write(line)
            f.write('\n')