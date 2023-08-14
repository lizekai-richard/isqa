# LLM_MRC_Summ

This is the repo for my UROP project. In this project, we improve LLM's summarization factuality by enhancing its machine reading comprehension ability.

## Zero-shot

To run zero-shot MRC experiment on SciMRC dataset, please run

```bash
bash scripts/run_scimrc.sh
```

To run zero-shot Summarization experiment on SciMRC/MUP dataset, please run

```bash
bash scripts/run_summ.sh
```

## Instruction-tune

To run the instruction-tuning experiment, please run

```bash
bash scripts/finetune.sh
```

### Inference

To test Vicuna's factuality of summarization after instruction-tuning, please run

```bash
bash scripts/summarization.sh
```

