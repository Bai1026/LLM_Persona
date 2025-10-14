# This folder is for all of the experiments for EMNLP 2025

<div align="center">

## 🌐 Project Website

### [**Visit Our Intro Website**](https://bai1026.github.io/LLM_Persona/)

_Explore BILLY's capabilities and learn about our research_

[![Website](https://img.shields.io/badge/Website-Intro_Page-blue?style=for-the-badge&logo=github-pages)](https://bai1026.github.io/LLM_Persona/)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2510.10157)

</div>

---

## How to run

- install requirements

  ```bash
  pip install -r gan_model.txt
  ```

- run the api of gan_model

  - At In_context_self_play folder

  ```bash
  python api.py
  ```

  - if wanna change the **Generator, Discriminator 討論輪數**, you can change `discussion_round` in the `./In_context_self_play/config/config.yaml` file
  - If u wanna customize the **Generator Prompt**, see this section (not using messages only)
    ```python
      gan_generator_prompt = gan_generator_1_en.format(system_message=system_message, conversation_list=conversation_list, last_user_message=last_user_message, discriminator_advice=discriminator_advice)
      # gan_generator_prompt = messages
    ```

- run the benchmark evaluation

  - At RPBench-Auto folder

  ```bash
  python run_character_eval.py -m1 gan_model
  ```

  - m1 here means the model 1
  - default model 2 is the baseline model (gpt-4o-mini we use here)
  - if wanna change the **evaluation 對話數**, you can change `line 13` in the `./RPBench-Auto/run_character_eval.py` file
  - if wanna change the **evaluation character**, you can change `line 14` in the `./RPBench-Auto/run_character_eval.py` file

- the result would appear in the `In_context_self_play/experimentx_log/gan` folder
  - the folder name would be the time of the experiment(vx for the x times of this day's experiment)

## In_context_self_play

- This folder contains the code of my implementation including api
  - Self-Play: This would be the baseline, generator with discriminator but the discriminator would not improve
  - GAN concept: This is the method we proposed, generator with discriminator and discriminator would improve simultaneously

## RPBench-Auto

- Evaluation benchmark for role-play
