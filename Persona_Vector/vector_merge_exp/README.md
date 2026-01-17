# ReadMe for Vectoer Merge Experiment

## Open the persona api u wanna use

- See `READMD_BAI.md` for further instruction.

## Vector Merge Experiment - Persona Trait Experiment

### Get the QA pair first

```bash
python vector_analysis.py
```

- would call default: `"http://localhost:5000/chat"`
- would use the neutral data in `persona_trati_data/neutral_task/dataset.json`

### Evaluate persona trait scores

```bash
python batch_persona_evaluator.py
```

- gotta choose type:
  - separate: each persona trait separately (0-100/each)
  - comprehensive: all persona traits together (0-100/all)
- Results would be saved in `persona_trait_results/neutral_task/Result`
- need to modify the files you wanna evaluate
  ```bash
      type_list = [
          'cre_env_fut_fut',
          'cre_env',
          'env_ana',
          'env',
          'baseline',
          'multi_prompt'
      ]
  ```
