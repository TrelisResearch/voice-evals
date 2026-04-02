#!/usr/bin/env python3
"""
Step 2: Download United-Syn-Med test.tar.gz, extract 50 random audio files,
build pilot dataset with transcriptions.
"""
import os, json, random, tarfile, io
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pathlib

HF_TOKEN = os.environ['HF_TOKEN']
random.seed(42)
OUT = pathlib.Path('medical-asr/phase1/tmp')
OUT.mkdir(exist_ok=True)
VOL = pathlib.Path('/mnt/HC_Volume_105102660/voice-evals-data')
VOL.mkdir(exist_ok=True)
AUDIO_OUT = VOL / 'united_audio'
AUDIO_OUT.mkdir(exist_ok=True)

# Load test split metadata (text only, fast)
print("Loading United-Syn-Med test metadata...")
ds = load_dataset('united-we-care/United-Syn-Med', split='test', token=HF_TOKEN)
print(f"Test rows: {len(ds)}, columns: {ds.column_names}")

# Sample 50 random rows
indices = random.sample(range(len(ds)), 50)
indices.sort()
sample_rows = ds.select(indices)
file_names = sample_rows['file_name']
transcriptions = sample_rows['transcription']

# Build lookup: filename → transcription
lookup = {fn: tx for fn, tx in zip(file_names, transcriptions)}
print(f"Sampled {len(lookup)} rows, e.g.:")
for fn, tx in list(lookup.items())[:3]:
    print(f"  {fn}: {tx[:80]}")

# Download test.tar.gz
print("\nDownloading test.tar.gz (4.4 GB)...")
tar_path = hf_hub_download(
    repo_id='united-we-care/United-Syn-Med',
    filename='data/audio/test.tar.gz',
    repo_type='dataset',
    token=HF_TOKEN,
    local_dir=str(VOL / 'united_cache'),
)
print(f"Downloaded to: {tar_path}")

# Extract only the 50 sampled files
print("Extracting 50 audio files from tar...")
extracted = {}
with tarfile.open(tar_path, 'r:gz') as tf:
    members = tf.getmembers()
    print(f"Tar contains {len(members)} files")
    for member in members:
        basename = os.path.basename(member.name)
        if basename in lookup:
            f = tf.extractfile(member)
            if f:
                audio_bytes = f.read()
                out_path = AUDIO_OUT / basename
                out_path.write_bytes(audio_bytes)
                extracted[basename] = str(out_path)
            if len(extracted) >= 50:
                break

print(f"Extracted {len(extracted)} audio files")
missing = set(lookup.keys()) - set(extracted.keys())
if missing:
    print(f"WARNING: {len(missing)} files not found in tar: {list(missing)[:5]}")

# Build pilot rows JSON
rows = []
for fn, tx in lookup.items():
    if fn in extracted:
        rows.append({
            'file_name': fn,
            'text': tx,
            'audio_path': extracted[fn],
            'source': 'united-we-care/United-Syn-Med',
        })

with open(OUT / 'united_pilot_rows.json', 'w') as f:
    json.dump(rows, f, indent=2)
print(f"\nSaved {len(rows)} rows to tmp/united_pilot_rows.json")

# Also save texts for LLM entity extraction
texts = [{'idx': i, 'text': r['text'], 'file_name': r['file_name']} for i, r in enumerate(rows)]
with open(OUT / 'united_pilot_texts.json', 'w') as f:
    json.dump(texts, f, indent=2)
print("Texts saved to tmp/united_pilot_texts.json")
print("\nStep 2 done.")
