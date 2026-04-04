#!/usr/bin/env python3
"""
Phase 2 Step 1: Mine (keyword, context) pairs from FDA DailyMed + PubMed.

Target: ~150 pairs across 6 medical categories:
  1. drugs_rare        - rare/difficult drug names (generics, biologics, chemo)
  2. procedures        - surgical + diagnostic procedures
  3. conditions_rare   - rare diseases + specific conditions
  4. anatomy           - anatomical Latin terms
  5. infectious        - pathogens, antivirals, antibiotics
  6. metabolic_endo    - metabolic/endocrine conditions + drugs

Output: phase2/tmp/pairs.json
"""
import os, json, time, requests
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

OUT = 'medical-asr/phase2/tmp/pairs.json'

# ─── Category definitions ────────────────────────────────────────────────────
CATEGORIES = {
    'drugs_rare': {
        'target': 35,
        'description': 'Rare/difficult drug names — biologics, chemotherapy, anticoagulants, specialty drugs',
        'pubmed_terms': [
            'pembrolizumab immunotherapy cancer',
            'tofacitinib rheumatoid arthritis JAK inhibitor',
            'apixaban anticoagulation atrial fibrillation',
            'eculizumab paroxysmal nocturnal hemoglobinuria',
            'dupilumab atopic dermatitis interleukin',
            'ixekizumab psoriasis treatment biologic',
            'sacubitril valsartan heart failure',
            'empagliflozin SGLT2 inhibitor diabetes',
            'bortezomib multiple myeloma proteasome',
            'nivolumab checkpoint inhibitor oncology',
            'venetoclax BCL-2 inhibitor leukemia',
            'liraglutide GLP-1 receptor agonist',
        ],
    },
    'procedures': {
        'target': 30,
        'description': 'Surgical and diagnostic procedures — complex names that ASR struggles with',
        'pubmed_terms': [
            'laparoscopic cholecystectomy bile duct',
            'endoscopic retrograde cholangiopancreatography ERCP',
            'transcatheter aortic valve replacement TAVR',
            'percutaneous coronary intervention stenting',
            'phacoemulsification cataract extraction',
            'thoracoscopic lobectomy pulmonary resection',
            'esophagogastroduodenoscopy Barrett esophagus',
            'fluorodeoxyglucose PET scan oncology',
            'electrophysiology catheter ablation arrhythmia',
        ],
    },
    'conditions_rare': {
        'target': 30,
        'description': 'Rare/specific disease names — ICD-10 range representation',
        'pubmed_terms': [
            'Guillain-Barré syndrome polyneuropathy',
            'Takayasu arteritis large vessel vasculitis',
            'sarcoidosis granulomatous inflammation',
            'systemic lupus erythematosus nephritis',
            'amyloidosis transthyretin cardiomyopathy',
            'myelodysplastic syndrome bone marrow',
            'pheochromocytoma catecholamine hypertension',
            'Zollinger-Ellison syndrome gastrinoma',
            'thrombotic thrombocytopenic purpura ADAMTS13',
        ],
    },
    'anatomy': {
        'target': 20,
        'description': 'Anatomical Latin and technical terms',
        'pubmed_terms': [
            'atrioventricular bundle His cardiac conduction',
            'substantia nigra dopaminergic neurons Parkinson',
            'glomerulosclerosis podocyte nephrotic syndrome',
            'hepatic sinusoid Kupffer cell liver',
            'trochanteric bursitis iliopsoas hip',
            'pterygoid muscle temporomandibular joint',
            'choledocholithiasis common bile duct',
        ],
    },
    'infectious': {
        'target': 20,
        'description': 'Infectious disease — pathogen names, antivirals, specific antibiotic regimens',
        'pubmed_terms': [
            'Clostridioides difficile vancomycin fidaxomicin',
            'Pseudomonas aeruginosa ceftazidime carbapenem',
            'Mycobacterium tuberculosis rifampicin isoniazid',
            'cytomegalovirus ganciclovir immunocompromised',
            'carbapenem-resistant Enterobacteriaceae colistin',
        ],
    },
    'metabolic_endo': {
        'target': 20,
        'description': 'Metabolic and endocrine conditions + drugs',
        'pubmed_terms': [
            'hypothyroidism levothyroxine TSH suppression',
            'Cushing syndrome hypercortisolism adrenal',
            'hyperparathyroidism cinacalcet calcimimetic',
            'phenylketonuria phenylalanine hydroxylase',
            'glycogen storage disease enzyme replacement',
        ],
    },
}

def search_pubmed(query, max_results=8):
    """Search PubMed and return abstracts."""
    base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'
    # Search
    r = requests.get(f'{base}/esearch.fcgi', params={
        'db': 'pubmed', 'term': query, 'retmax': max_results,
        'retmode': 'json', 'sort': 'relevance',
    }, timeout=15)
    ids = r.json().get('esearchresult', {}).get('idlist', [])
    if not ids:
        return []
    time.sleep(0.34)  # NCBI rate limit: 3 req/s

    # Fetch abstracts
    r2 = requests.get(f'{base}/efetch.fcgi', params={
        'db': 'pubmed', 'id': ','.join(ids[:5]),
        'rettype': 'abstract', 'retmode': 'text',
    }, timeout=20)
    time.sleep(0.34)
    return r2.text.split('\n\n')

def extract_sentences_with_keyword(text_blocks, keyword):
    """Find sentences containing the keyword from PubMed text."""
    import re
    keyword_lower = keyword.lower()
    sentences = []
    for block in text_blocks:
        # Split into sentences roughly
        sents = re.split(r'(?<=[.!?])\s+', block)
        for s in sents:
            s = s.strip()
            if len(s) < 40 or len(s) > 300:
                continue
            if keyword_lower in s.lower():
                # Clean up PubMed formatting artifacts
                s = re.sub(r'\s+', ' ', s)
                s = re.sub(r'^\d+\.\s*', '', s)
                sentences.append(s)
    return sentences

def mine_category(cat_name, cat_config):
    """Mine pairs for one category."""
    pairs = []
    seen_keywords = set()

    for term in cat_config['pubmed_terms']:
        # The first word of the term is usually the keyword
        keyword = term.split()[0].rstrip(',')
        if keyword.lower() in seen_keywords:
            continue

        print(f'  [{cat_name}] searching: {term[:60]}...')
        try:
            blocks = search_pubmed(term, max_results=5)
            sents = extract_sentences_with_keyword(blocks, keyword)
            if sents:
                # Pick best sentence: prefer moderate length, medical density
                sents_sorted = sorted(sents, key=lambda s: abs(len(s) - 120))
                context = sents_sorted[0]
                pairs.append({
                    'keyword': keyword,
                    'category': cat_name,
                    'context': context,
                    'source': f'PubMed: {term}',
                    'description': cat_config['description'],
                })
                seen_keywords.add(keyword.lower())
                print(f'    → found: {keyword!r}: {context[:80]}...')
            else:
                print(f'    → no sentences found for {keyword!r}')
        except Exception as e:
            print(f'    → error: {e}')

        if len(pairs) >= cat_config['target'] // 2:
            break  # enough from PubMed; FDA will supply more for drugs

    return pairs

def mine_fda_drugs(target=35):
    """Mine rare drug names + context from FDA drug labels via OpenFDA API."""
    pairs = []
    seen = set()

    # Search for specialty drug labels with complex names
    searches = [
        'openfda.pharm_class_epc:"Tyrosine Kinase Inhibitor"',
        'openfda.pharm_class_epc:"Monoclonal Antibody"',
        'openfda.pharm_class_epc:"Proteasome Inhibitor"',
        'openfda.pharm_class_epc:"Checkpoint Inhibitor"',
        'openfda.pharm_class_epc:"JAK Inhibitor"',
        'openfda.pharm_class_epc:"SGLT2 Inhibitor"',
        'openfda.pharm_class_epc:"GLP-1 Receptor Agonist"',
        'openfda.pharm_class_epc:"CDK Inhibitor"',
    ]

    for search in searches:
        if len(pairs) >= target:
            break
        try:
            r = requests.get('https://api.fda.gov/drug/label.json', params={
                'search': search,
                'limit': 5,
            }, timeout=15)
            results = r.json().get('results', [])
            time.sleep(0.5)

            for label in results:
                # Get brand + generic name
                openfda = label.get('openfda', {})
                brand = (openfda.get('brand_name') or [''])[0]
                generic = (openfda.get('generic_name') or [''])[0]
                keyword = generic.split()[0] if generic else brand.split()[0]
                if not keyword or keyword.lower() in seen or len(keyword) < 6:
                    continue

                # Extract a sentence from indications_and_usage
                indications = label.get('indications_and_usage', [''])[0]
                if not indications:
                    continue

                import re
                sents = re.split(r'(?<=[.!?])\s+', indications)
                good = [s.strip() for s in sents
                        if 40 < len(s.strip()) < 280
                        and keyword.lower() in s.lower()]
                if not good:
                    # Use first decent sentence
                    good = [s.strip() for s in sents if 60 < len(s.strip()) < 250][:1]
                if not good:
                    continue

                context = good[0]
                pairs.append({
                    'keyword': keyword,
                    'category': 'drugs_rare',
                    'context': context,
                    'source': f'FDA label: {brand} ({generic})',
                    'description': 'Rare/difficult drug names',
                })
                seen.add(keyword.lower())
                print(f'  [drugs_rare/FDA] {keyword}: {context[:80]}...')

                if len(pairs) >= target:
                    break

        except Exception as e:
            print(f'  FDA search error: {e}')

    return pairs

# ── Main ─────────────────────────────────────────────────────────────────────
all_pairs = []

print('=== Mining FDA drug labels ===')
fda_pairs = mine_fda_drugs(target=25)
all_pairs.extend(fda_pairs)
print(f'FDA: {len(fda_pairs)} pairs')

print('\n=== Mining PubMed abstracts ===')
for cat_name, cat_config in CATEGORIES.items():
    cat_pairs = mine_category(cat_name, cat_config)
    all_pairs.extend(cat_pairs)
    print(f'{cat_name}: {len(cat_pairs)} pairs (total so far: {len(all_pairs)})')
    time.sleep(1)

# Deduplicate by keyword
seen_kw = set()
deduped = []
for p in all_pairs:
    k = p['keyword'].lower()
    if k not in seen_kw:
        seen_kw.add(k)
        deduped.append(p)

print(f'\n=== Total: {len(deduped)} unique pairs ===')

# Category breakdown
from collections import Counter
cats = Counter(p['category'] for p in deduped)
for cat, n in sorted(cats.items()):
    print(f'  {cat}: {n}')

json.dump(deduped, open(OUT, 'w'), indent=2)
print(f'\nSaved to {OUT}')
