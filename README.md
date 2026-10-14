# Whisper-small LoRA for Hindi ASR + articulatory error analysis

Parameter-efficient fine-tuning of Whisper-small on Hindi (FLEURS), plus a
character-aligned analysis of **which articulatory contrasts the model gets wrong**
(aspiration, voicing, retroflex vs dental place, vowel length) before and after
fine-tuning.

Everything was run on a free Kaggle T4 (single GPU, fp16).

## Results (FLEURS `hi_in` test, 417 utterances)

| Model | WER ↓ | CER ↓ | Trainable params |
|---|---|---|---|
| Whisper-small, zero-shot | 78.1 | 44.3 | 0 |
| Whisper-small + LoRA (r=32) | **26.5** | **9.5** | 7.08M (2.84%) |

WER/CER use NFC normalisation with punctuation removed (see `src/articulatory.py`).
<!-- TODO: add 95% CIs from results/wer_cer_ci.csv after running src/bootstrap_ci.py -->

### Error analysis: the model collapses the marked member of a contrast

% of reference characters replaced by their contrast partner (point estimates; test set only):

| Direction | Zero-shot | LoRA |
|---|---|---|
| aspirated → unaspirated | 31.3 | 8.1 |
| unaspirated → aspirated | 1.5 | 0.7 |
| retroflex → dental | 17.7 | 3.1 |
| dental → retroflex | 2.0 | 0.2 |
| voiced → voiceless | 3.6 | 2.7 |
| voiceless → voiced | 2.5 | 0.7 |
| long → short vowel | 12.0 | 4.8 |
| short → long vowel | 8.6 | 3.4 |

Fine-tuning shrinks every error type, but the asymmetry remains: after LoRA,
aspirated→unaspirated is ~11x more frequent than the reverse, and retroflex→dental ~14x.
Vowel length is close to symmetric.
Full per-class tables: `results/class_errors.csv`, `results/contrast_errors.csv`.
<!-- TODO: add bootstrap CIs (results/asymmetry_ci.csv) and results/contrast_errors_ci.png -->

### Limitations (please read)
- **Grapheme-level proxy, not phones.** Devanagari is largely phonemic, but spelling
  variants (nukta, anusvara vs nasal conjunct) count as errors. The
  "nasalisation/nukta/virama" class is mostly orthographic and should not be read as acoustic.
- **Small counts for some contrasts** (e.g. retroflex→dental after LoRA is ~28 events).
  Use the bootstrap CIs before interpreting ratios.
- **Single seed, single test set, clean read speech.** No dysarthric / noisy / accented evaluation yet.
- Contrast-partner swaps are a minority of all errors; most zero-shot errors are general garbling.

## Repository layout

```
src/train_and_eval.py     # data caching, zero-shot baseline, LoRA / full fine-tune, WER/CER, error analysis
src/articulatory.py       # Devanagari taxonomy, normalisation, character alignment
src/bootstrap_ci.py       # utterance-level bootstrap CIs (CPU only)
notebooks/                # the Kaggle notebook of the reported run
results/                  # CSVs and test hypotheses from the reported run
```

## Reproduce

**Training + evaluation (Kaggle T4, Internet on):**
1. Upload `src/train_and_eval.py` (or paste its `# %%` cells into a notebook).
2. Run cell 1, restart the kernel, run the remaining cells.
3. Config is at the top of cell 2: `MODE="lora"` (Whisper-small) or `MODE="full"` with
   `MODEL_ID="openai/whisper-base"`.

Outputs land in `/kaggle/working/<run>/` (`summary.csv`, `contrast_errors.csv`,
`class_errors.csv`, `test_hypotheses.json`, `test_references.json`).

**Bootstrap confidence intervals (local, no GPU):**
```bash
pip install -r requirements.txt
cd src
python bootstrap_ci.py --results ../results --n-boot 5000
```
This needs `results/test_references.json`. Runs made with the current
`train_and_eval.py` write it automatically; for the reported run, export it once in the
Kaggle notebook:
```python
import json
json.dump(list(ds["test"]["text"]), open(f"{OUT}/test_references.json", "w"),
          ensure_ascii=False, indent=1)
```

## Method notes
- Data: FLEURS Hindi (`google/fleurs`, `hi_in`): 2115 train / 239 val / 417 test after
  filtering utterances over 30 s or over 448 label tokens.
- LoRA on `q_proj, k_proj, v_proj, out_proj`, r=32, alpha=64, dropout 0.05; lr 1e-3, 8 epochs,
  effective batch 32, fp16, gradient checkpointing; best checkpoint by validation WER; adapter merged for evaluation.
- Error analysis: reference and hypothesis (spaces removed) are aligned with Levenshtein
  distance; for every reference character belonging to a contrast we count whether it was
  replaced by its partner (e.g. ख→क for aspirated→unaspirated).
- Bootstrap: resample utterances with replacement (same resamples for all models, so
  model comparisons are paired).

## License
MIT
