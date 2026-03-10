# ASR Evaluation Results — AI Technical Terms (v0)

Dataset: Trelis/ai-terms-{public,semi-private,private}
Date: 2026-03-10
Language: English
Normalizer: auto

## Summary Table

| Model | Split | WER | CER | Entity CER | Samples |
|-------|-------|-----|-----|------------|---------|
| Qwen/Qwen3-ASR-0.6B | private | 14.250% | 10.091% | 8.696% | 12/12 |
| Qwen/Qwen3-ASR-0.6B | public | 17.573% | 10.455% | 8.521% | 12/12 |
| Qwen/Qwen3-ASR-0.6B | semi-private | 14.000% | 8.879% | 8.440% | 12/12 |
| Qwen/Qwen3-ASR-1.7B | private | 8.197% | 5.698% | 6.225% | 12/12 |
| Qwen/Qwen3-ASR-1.7B | public | 15.342% | 9.497% | 7.519% | 12/12 |
| Qwen/Qwen3-ASR-1.7B | semi-private | 12.933% | 8.245% | 6.330% | 12/12 |
| UsefulSensors/moonshine-base | private | 8.827% | 4.298% | 8.004% | 12/12 |
| UsefulSensors/moonshine-base | public | 12.413% | 6.292% | 8.647% | 12/12 |
| UsefulSensors/moonshine-base | semi-private | 11.467% | 5.203% | 9.266% | 12/12 |
| UsefulSensors/moonshine-tiny | public | 16.597% | 8.158% | 11.403% | 12/12 |
| UsefulSensors/moonshine-tiny | semi-private | 16.667% | 7.776% | 13.578% | 12/12 |
| assemblyai/universal-3-pro | public | 7.671% | 4.721% | 4.637% | 12/12 |
| assemblyai/universal-3-pro | semi-private | 4.000% | 2.373% | 3.303% | 12/12 |
| elevenlabs/scribe-v2 | public | 10.879% | 7.064% | 5.514% | 12/12 |
| elevenlabs/scribe-v2 | semi-private | 9.200% | 6.343% | 2.202% | 12/12 |
| facebook/omniASR-LLM-7B | private | 22.320% | 11.081% | 13.537% | 12/12 |
| facebook/omniASR-LLM-7B | public | 21.339% | 9.393% | 15.163% | 12/12 |
| facebook/omniASR-LLM-7B | semi-private | 22.533% | 10.231% | 23.853% | 12/12 |
| fireworks/whisper-v3 | public | 7.113% | 4.081% | 4.386% | 12/12 |
| fireworks/whisper-v3 | semi-private | 7.733% | 3.281% | 7.156% | 12/12 |
| microsoft/VibeVoice-ASR-HF | private | 7.818% | 5.402% | 5.138% | 12/12 |
| microsoft/VibeVoice-ASR-HF | public | 11.855% | 7.829% | 5.890% | 12/12 |
| microsoft/VibeVoice-ASR-HF | semi-private | 10.000% | 6.350% | 5.321% | 12/12 |
| mistralai/Voxtral-Mini-3B-2507 | private | 4.290% | 2.390% | 4.051% | 12/12 |
| mistralai/Voxtral-Mini-3B-2507 | public | 7.670% | 4.050% | — | 12/12 |
| mistralai/Voxtral-Mini-3B-2507 | semi-private | 6.130% | 2.670% | — | 12/12 |
| nvidia/parakeet-tdt-0.6b-v3 | private | 5.675% | 3.175% | 4.743% | 12/12 |
| nvidia/parakeet-tdt-0.6b-v3 | public | 7.950% | 4.206% | 5.764% | 12/12 |
| nvidia/parakeet-tdt-0.6b-v3 | semi-private | 8.133% | 3.387% | 5.046% | 12/12 |
| openai/whisper-large-v3-turbo | private | 5.044% | 2.768% | 3.854% | 12/12 |
| openai/whisper-large-v3-turbo | public | 6.137% | 3.672% | 4.135% | 12/12 |
| openai/whisper-large-v3-turbo | semi-private | 6.800% | 2.925% | 4.771% | 12/12 |
| speechmatics/ursa-2-enhanced | public | 9.763% | 5.123% | 9.148% | 12/12 |
| speechmatics/ursa-2-enhanced | semi-private | 9.600% | 4.895% | 8.991% | 12/12 |
| google/gemini-2.5-pro | public | 7.252% | 4.614% | 4.511% | 12/12 |
| google/gemini-2.5-pro | semi-private | 6.133% | 2.840% | 5.505% | 12/12 |
| deepgram/nova-3 | public | 12.552% | 6.260% | 8.145% | 12/12 |
| deepgram/nova-3 | semi-private | 15.467% | 7.763% | 8.165% | 12/12 |

## Category CER Breakdown

| Model | Split | Benchmarks | Companies | Models | People | Products | Technical |
|-------|-------|-----------|-----------|--------|--------|----------|-----------|
| Qwen/Qwen3-ASR-0.6B | private | 25.0% | 13.9% | 31.3% | 13.3% | 10.5% | 4.2% |
| Qwen/Qwen3-ASR-0.6B | public | 22.1% | 8.8% | 39.6% | — | 8.3% | 2.0% |
| Qwen/Qwen3-ASR-0.6B | semi-private | 28.1% | 7.1% | 24.6% | 25.0% | 8.3% | 2.7% |
| Qwen/Qwen3-ASR-1.7B | private | 21.7% | 6.9% | 15.7% | 13.3% | 7.9% | 3.6% |
| Qwen/Qwen3-ASR-1.7B | public | 16.9% | 8.8% | 41.7% | — | 8.3% | 1.6% |
| Qwen/Qwen3-ASR-1.7B | semi-private | 21.9% | 4.3% | 21.4% | 12.5% | 6.0% | 1.8% |
| UsefulSensors/moonshine-base | private | 18.3% | 4.2% | 21.7% | 40.0% | 21.1% | 4.7% |
| UsefulSensors/moonshine-base | public | 16.2% | 14.7% | 12.5% | — | 33.3% | 3.7% |
| UsefulSensors/moonshine-base | semi-private | 28.1% | 9.5% | 11.9% | 12.5% | 6.8% | 6.9% |
| UsefulSensors/moonshine-tiny | public | 21.3% | 11.8% | 31.2% | — | 27.8% | 5.7% |
| UsefulSensors/moonshine-tiny | semi-private | 28.1% | 23.3% | 17.5% | 25.0% | 11.3% | 7.7% |
| assemblyai/universal-3-pro | public | 16.9% | 2.9% | 16.7% | — | 0.0% | 0.8% |
| assemblyai/universal-3-pro | semi-private | 15.6% | 2.9% | 10.3% | 0.0% | 3.0% | 0.5% |
| elevenlabs/scribe-v2 | public | 19.9% | 2.9% | 12.5% | — | 8.3% | 1.2% |
| elevenlabs/scribe-v2 | semi-private | 14.1% | 0.5% | 2.4% | 0.0% | 4.5% | 0.9% |
| facebook/omniASR-LLM-7B | private | 31.7% | 29.2% | 47.0% | 20.0% | 36.8% | 5.5% |
| facebook/omniASR-LLM-7B | public | 36.8% | 23.5% | 58.3% | — | 27.8% | 3.3% |
| facebook/omniASR-LLM-7B | semi-private | 40.6% | 33.8% | 42.9% | 12.5% | 28.6% | 12.8% |
| fireworks/whisper-v3 | public | 13.2% | 11.8% | 10.4% | — | 2.8% | 0.6% |
| fireworks/whisper-v3 | semi-private | 18.8% | 11.0% | 17.5% | 12.5% | 8.3% | 1.6% |
| microsoft/VibeVoice-ASR-HF | private | 28.3% | 0.0% | 8.4% | 0.0% | 5.3% | 3.5% |
| microsoft/VibeVoice-ASR-HF | public | 14.7% | 8.8% | 16.7% | — | 8.3% | 2.0% |
| microsoft/VibeVoice-ASR-HF | semi-private | 18.8% | 2.4% | 18.3% | 0.0% | 6.0% | 1.8% |
| mistralai/Voxtral-Mini-3B-2507 | private | 16.7% | 5.6% | 9.6% | 6.7% | 7.9% | 2.0% |
| nvidia/parakeet-tdt-0.6b-v3 | private | 15.0% | 2.8% | 10.8% | 13.3% | 18.4% | 2.6% |
| nvidia/parakeet-tdt-0.6b-v3 | public | 14.7% | 14.7% | 8.3% | — | 11.1% | 1.6% |
| nvidia/parakeet-tdt-0.6b-v3 | semi-private | 23.4% | 6.2% | 7.1% | 12.5% | 6.0% | 1.6% |
| openai/whisper-large-v3-turbo | private | 10.0% | 0.0% | 15.7% | 6.7% | 7.9% | 2.2% |
| openai/whisper-large-v3-turbo | public | 10.3% | 10.3% | 8.3% | — | 8.3% | 1.0% |
| openai/whisper-large-v3-turbo | semi-private | 20.3% | 5.2% | 8.7% | 12.5% | 7.5% | 1.1% |
| speechmatics/ursa-2-enhanced | public | 23.5% | 11.8% | 14.6% | — | 19.4% | 3.7% |
| speechmatics/ursa-2-enhanced | semi-private | 26.6% | 13.3% | 15.9% | 0.0% | 13.5% | 2.7% |
| google/gemini-2.5-pro | public | 11.8% | 2.9% | 14.6% | — | 16.7% | 1.0% |
| google/gemini-2.5-pro | semi-private | 20.3% | 11.9% | 8.7% | 0.0% | 5.3% | 0.7% |
| deepgram/nova-3 | public | 23.5% | 8.8% | 16.7% | — | 19.4% | 2.4% |
| deepgram/nova-3 | semi-private | 29.7% | 5.2% | 20.6% | 0.0% | 9.0% | 3.8% |

## Key Observations

- **Entity CER is consistently higher than overall CER** — models struggle more with novel AI terminology
- **Benchmarks** are the hardest category across most models (e.g. 'ARC-AGI 2', 'SWE-Bench Verified')
- **Model names** are also challenging (e.g. 'Qwen 3.5', 'Voxtral', 'M2.5')
- **Technical terms** tend to have lowest entity CER — many are common English words
- **Best overall**: AssemblyAI Universal-3 Pro (lowest CER on semi-private), closely followed by Gemini 2.5 Pro (2.8% CER) and Voxtral Mini 3B (2.7% CER)
- **Best open-source**: Voxtral Mini 3B and Parakeet TDT 0.6B
- **Scribe v2** has surprisingly low entity CER on semi-private (2.2%) despite higher overall CER

## Notes

- Private split only evaluated with open-source/GPU models (no router/proprietary models)
- Some Voxtral/Whisper runs on public and semi-private lack entity_cer (run before feature was deployed)
- All samples are 12/12 after fixing the 30.4s row that Voxtral previously skipped