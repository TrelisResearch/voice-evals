#!/usr/bin/env python3
"""Quick TTS test: 3 sentences, af_heart, log full responses."""
import os, json, time, requests
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

API_KEY = os.environ['TRELIS_STUDIO_API_KEY']
BASE    = 'https://studio.trelis.com/api/v1'
H       = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}

t0 = time.time()
def elapsed(): return f'{(time.time()-t0):.0f}s'

SENTENCES_FILE = 'medical-asr/phase2/tmp/sentences.json'
sentences = json.load(open(SENTENCES_FILE))[:3]
texts = [s['tts_text'] for s in sentences]
print(f'Testing TTS with {len(texts)} sentences:')
for t in texts:
    print(f'  {t[:80]}')

# Step 1: Get upload URLs
print(f'\n[{elapsed()}] Getting upload URLs...')
filenames = [f'test_{i:03d}.txt' for i in range(len(texts))]
text_bytes = [t.encode('utf-8') for t in texts]
r = requests.post(f'{BASE}/file-stores/upload-urls', headers=H, json={
    'files': [
        {'filename': fn, 'size_bytes': len(tb), 'content_type': 'text/plain'}
        for fn, tb in zip(filenames, text_bytes)
    ],
    'name': 'medical-terms-tts-test',
})
print(f'  status={r.status_code}')
resp = r.json()
print(f'  response keys: {list(resp.keys())}')
store_id = resp.get('file_store_id')
upload_entries = resp.get('files', [])
print(f'  store_id={store_id}, {len(upload_entries)} entries')

# Step 2: Upload .txt files
print(f'\n[{elapsed()}] Uploading files...')
for entry, tb in zip(upload_entries, text_bytes):
    url = entry['upload_url']
    ct = entry.get('content_type', 'text/plain')
    put_r = requests.put(url, data=tb, headers={'Content-Type': ct}, timeout=30)
    print(f'  {entry["filename"]}: {put_r.status_code}')

# Step 3: Submit TTS job
print(f'\n[{elapsed()}] Submitting TTS job (kokoro/af_heart)...')
r = requests.post(f'{BASE}/file-stores/{store_id}/tts', headers=H, json={
    'engine': 'kokoro',
    'kokoro_voice': 'af_heart',
})
print(f'  status={r.status_code}')
tts_resp = r.json()
print(f'  response: {json.dumps(tts_resp, indent=2)[:500]}')
tts_job_id = tts_resp.get('job_id') or tts_resp.get('id')
print(f'  job_id={tts_job_id}')

# Step 4: Poll
print(f'\n[{elapsed()}] Polling TTS job {tts_job_id}...')
for i in range(120):
    r = requests.get(f'{BASE}/data-prep/jobs/{tts_job_id}', headers=H)
    data = r.json()
    status = data.get('status', '?')
    print(f'  [{elapsed()}] {status}', flush=True)
    if status in ('completed', 'done', 'ready'):
        print(f'\nFull job response:')
        print(json.dumps(data, indent=2)[:2000])
        output_store = (data.get('result') or {}).get('output_file_store_id') or data.get('output_file_store_id')
        print(f'\noutput_store_id={output_store}')
        break
    elif status in ('failed', 'error', 'aborted'):
        print(f'FAILED:\n{json.dumps(data, indent=2)[:1000]}')
        break
    time.sleep(15)

print(f'\n[{elapsed()}] DONE')
