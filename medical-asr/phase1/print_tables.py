import json

with open('medical-asr/phase1/tmp/eval_results.json') as f:
    results = json.load(f)

completed = [r for r in results if r['status'] == 'completed']

DS_LABELS = {'eka': 'EKA (real Indian clinical)', 'multimed': 'MultiMed EN (lecture/podcast)', 'united': 'United-Syn-Med (synthetic drug)'}

for ds in ['eka', 'multimed', 'united']:
    rows = sorted([r for r in completed if r['dataset'] == ds], key=lambda x: x.get('entity_cer') or 1.0)
    print(f'\n### {DS_LABELS[ds]}')
    print(f'{"Model":<38} {"WER":>6} {"CER":>6} {"EntCER":>8}')
    print('-' * 62)
    for r in rows:
        wer = f"{r['wer']:.3f}" if r['wer'] is not None else '  n/a'
        cer = f"{r['cer']:.3f}" if r['cer'] is not None else '  n/a'
        ecer = f"{r['entity_cer']:.3f}" if r['entity_cer'] is not None else '  n/a'
        model = r['model'].split('/')[-1]
        print(f'{model:<38} {wer:>6} {cer:>6} {ecer:>8}')

# EKA category breakdown
print('\n\n### EKA — Category CER breakdown')
eka_rows = sorted([r for r in completed if r['dataset'] == 'eka'], key=lambda x: x.get('entity_cer') or 1.0)
cats = ['drugs', 'clinical_findings', 'diagnostics', 'advices', 'misc_medical']
header = f'  {"Model":<30}' + ''.join(f'  {c[:8]:>8}' for c in cats)
print(header)
print('  ' + '-' * 74)
for r in eka_rows:
    cc = r.get('category_cer', {})
    if not cc:
        continue
    model = r['model'].split('/')[-1][:30]
    vals = ''.join(f'  {cc.get(c, 0):>8.3f}' for c in cats)
    print(f'  {model:<30}{vals}')
