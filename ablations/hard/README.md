# Alternativas de remoção de questões fáceis

Resultados das regras exploratórias sobre a base deduplicada. A versão principal é MMLU-PT-Hard-Direct v2 (`reference3`).

Para cada questão e modelo: média = (extractive_match_0shot + extractive_match_5shot)/2. Ambos os protocolos participam da seleção.

Painel de referência: gemini-3.8-flash; google/gemma-4-31B-it; Qwen/Qwen3.5-27B.
Painel compacto: Polygl0t/Tucano2-qwen-0.5B-Instruct; Qwen/Qwen3.5-0.8B; google/gemma-4-E2B-it; meta-llama/Llama-3.2-1B-Instruct; nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16.

Grupos com pesos iguais: Polygl0t: Polygl0t/Tucano2-qwen-0.5B-Instruct, Polygl0t/Tucano2-qwen-1.5B-Instruct, Polygl0t/Tucano2-qwen-3.7B-Instruct; Qwen: Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-27B, Qwen/Qwen3.5-2B, Qwen/Qwen3.5-35B-A3B, Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B; gemini: gemini-2.5-flash, gemini-3.8-flash; google: google/gemma-4-12B-it, google/gemma-4-26B-A4B-it, google/gemma-4-31B-it, google/gemma-4-E2B-it, google/gemma-4-E4B-it; meta-llama: meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B-Instruct; nvidia: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16, nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16.
Grupos operacionais por série/fornecedor, não famílias independentes: Tucano2-qwen compartilha ancestralidade com Qwen.

## Regras

| Identificador | Remover quando |
| --- | --- |
| original | Sem remoção |
| all21 | Remover soma das médias =21: 42/42 acertos |
| atleast19 | Remover soma das médias >=19: >=38/42 acertos (90,48%) |
| atleast17 | Remover soma das médias >=17: >=34/42 acertos (80,95%) |
| families80 | Remover média >=80% com peso igual por grupo |
| compact4 | Remover soma das médias dos compactos >=4: >=8/10 acertos |
| compact3 | Remover soma das médias dos compactos >=3: >=6/10 acertos |
| reference3 | Remover soma das médias do painel =3: 6/6 acertos |

## Contagens

| Regra | Nível | Original | Mantidas | Removidas | % removida |
| --- | --- | ---: | ---: | ---: | ---: |
| original | high_school | 8704 | 8704 | 0 | 0.00 |
| original | undergraduate | 29094 | 29094 | 0 | 0.00 |
| all21 | high_school | 8704 | 8407 | 297 | 3.41 |
| all21 | undergraduate | 29094 | 28594 | 500 | 1.72 |
| atleast19 | high_school | 8704 | 7193 | 1511 | 17.36 |
| atleast19 | undergraduate | 29094 | 25380 | 3714 | 12.77 |
| atleast17 | high_school | 8704 | 6422 | 2282 | 26.22 |
| atleast17 | undergraduate | 29094 | 21781 | 7313 | 25.14 |
| families80 | high_school | 8704 | 6457 | 2247 | 25.82 |
| families80 | undergraduate | 29094 | 22091 | 7003 | 24.07 |
| compact4 | high_school | 8704 | 7240 | 1464 | 16.82 |
| compact4 | undergraduate | 29094 | 24460 | 4634 | 15.93 |
| compact3 | high_school | 8704 | 6176 | 2528 | 29.04 |
| compact3 | undergraduate | 29094 | 19953 | 9141 | 31.42 |
| reference3 | high_school | 8704 | 4584 | 4120 | 47.33 |
| reference3 | undergraduate | 29094 | 10684 | 18410 | 63.28 |

## high_school — média 0/5-shot

Acurácia micro (%).

| Modelo | original | all21 | atleast19 | atleast17 | families80 | compact4 | compact3 | reference3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Polygl0t/Tucano2-qwen-0.5B-Instruct | 34.82 | 32.52 | 24.78 | 21.75 | 21.81 | 24.12 | 19.11 | 21.42 |
| Polygl0t/Tucano2-qwen-1.5B-Instruct | 40.50 | 38.40 | 30.27 | 25.51 | 25.49 | 31.84 | 26.52 | 23.19 |
| Polygl0t/Tucano2-qwen-3.7B-Instruct | 47.37 | 45.51 | 37.84 | 32.16 | 32.41 | 39.21 | 33.36 | 23.91 |
| Qwen/Qwen3.5-0.8B | 37.57 | 35.37 | 27.28 | 24.35 | 24.97 | 27.12 | 21.36 | 24.51 |
| Qwen/Qwen3.5-27B | 69.20 | 68.11 | 62.73 | 58.30 | 58.53 | 63.25 | 58.52 | 41.51 |
| Qwen/Qwen3.5-2B | 45.33 | 43.40 | 34.46 | 28.75 | 29.41 | 35.66 | 29.09 | 25.68 |
| Qwen/Qwen3.5-35B-A3B | 65.60 | 64.38 | 58.39 | 53.58 | 53.87 | 59.03 | 53.76 | 39.33 |
| Qwen/Qwen3.5-4B | 57.26 | 55.75 | 48.43 | 42.68 | 43.12 | 49.34 | 43.12 | 32.49 |
| Qwen/Qwen3.5-9B | 60.58 | 59.19 | 52.33 | 46.90 | 47.31 | 53.18 | 47.41 | 33.84 |
| gemini-2.5-flash | 68.61 | 67.50 | 62.05 | 57.63 | 57.81 | 62.68 | 58.06 | 43.35 |
| gemini-3.8-flash | 90.71 | 90.38 | 88.79 | 87.50 | 87.55 | 89.02 | 87.79 | 82.35 |
| google/gemma-4-12B-it | 56.55 | 55.02 | 47.50 | 41.72 | 42.18 | 48.49 | 42.93 | 25.91 |
| google/gemma-4-26B-A4B-it | 60.15 | 58.74 | 51.87 | 46.52 | 46.88 | 52.82 | 47.77 | 29.56 |
| google/gemma-4-31B-it | 63.25 | 61.95 | 55.63 | 50.51 | 50.88 | 56.43 | 51.62 | 30.22 |
| google/gemma-4-E2B-it | 44.90 | 42.95 | 33.78 | 27.59 | 28.23 | 34.23 | 26.11 | 22.00 |
| google/gemma-4-E4B-it | 54.37 | 52.76 | 44.86 | 38.62 | 38.99 | 45.78 | 39.29 | 26.54 |
| meta-llama/Llama-3.1-8B-Instruct | 48.79 | 46.98 | 38.43 | 32.30 | 32.61 | 39.85 | 33.48 | 26.40 |
| meta-llama/Llama-3.2-1B-Instruct | 32.07 | 29.67 | 24.28 | 22.84 | 22.75 | 23.05 | 19.33 | 24.35 |
| meta-llama/Llama-3.2-3B-Instruct | 42.92 | 40.91 | 31.89 | 26.74 | 26.81 | 33.30 | 27.40 | 25.46 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 53.10 | 51.45 | 43.42 | 37.45 | 37.48 | 44.45 | 38.24 | 30.49 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 46.90 | 45.03 | 36.28 | 30.42 | 30.11 | 36.54 | 28.39 | 26.25 |

## high_school — 0-shot

Acurácia micro (%).

| Modelo | original | all21 | atleast19 | atleast17 | families80 | compact4 | compact3 | reference3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Polygl0t/Tucano2-qwen-0.5B-Instruct | 33.48 | 31.13 | 23.54 | 20.60 | 20.66 | 22.79 | 17.68 | 19.92 |
| Polygl0t/Tucano2-qwen-1.5B-Instruct | 37.86 | 35.66 | 28.22 | 24.20 | 24.05 | 29.81 | 24.77 | 22.40 |
| Polygl0t/Tucano2-qwen-3.7B-Instruct | 42.64 | 40.61 | 33.31 | 28.04 | 28.25 | 34.75 | 29.21 | 20.68 |
| Qwen/Qwen3.5-0.8B | 37.17 | 34.95 | 27.43 | 24.65 | 25.34 | 27.33 | 21.44 | 25.04 |
| Qwen/Qwen3.5-27B | 68.85 | 67.75 | 62.32 | 57.85 | 58.08 | 62.83 | 58.06 | 40.86 |
| Qwen/Qwen3.5-2B | 44.52 | 42.56 | 33.42 | 27.50 | 28.26 | 34.61 | 27.98 | 24.11 |
| Qwen/Qwen3.5-35B-A3B | 65.97 | 64.77 | 58.85 | 54.11 | 54.39 | 59.43 | 54.21 | 40.03 |
| Qwen/Qwen3.5-4B | 57.03 | 55.51 | 48.17 | 42.39 | 42.85 | 49.14 | 42.78 | 32.35 |
| Qwen/Qwen3.5-9B | 61.60 | 60.25 | 53.55 | 48.30 | 48.69 | 54.41 | 48.75 | 35.67 |
| gemini-2.5-flash | 64.32 | 63.05 | 56.87 | 51.92 | 52.11 | 57.60 | 52.38 | 36.04 |
| gemini-3.8-flash | 91.68 | 91.39 | 89.98 | 88.84 | 88.86 | 90.18 | 89.12 | 84.21 |
| google/gemma-4-12B-it | 51.59 | 49.88 | 41.44 | 34.93 | 35.40 | 42.46 | 36.20 | 17.87 |
| google/gemma-4-26B-A4B-it | 55.50 | 53.93 | 46.23 | 40.17 | 40.59 | 47.22 | 41.52 | 21.29 |
| google/gemma-4-31B-it | 57.18 | 55.67 | 48.27 | 42.18 | 42.60 | 49.10 | 43.33 | 18.70 |
| google/gemma-4-E2B-it | 43.05 | 41.04 | 31.60 | 25.16 | 25.83 | 32.10 | 23.98 | 18.43 |
| google/gemma-4-E4B-it | 52.47 | 50.79 | 42.56 | 36.11 | 36.52 | 43.56 | 36.87 | 23.28 |
| meta-llama/Llama-3.1-8B-Instruct | 48.84 | 47.03 | 38.51 | 32.54 | 32.83 | 39.90 | 33.57 | 27.12 |
| meta-llama/Llama-3.2-1B-Instruct | 32.23 | 29.83 | 24.57 | 22.95 | 22.94 | 23.18 | 19.40 | 24.83 |
| meta-llama/Llama-3.2-3B-Instruct | 43.46 | 41.47 | 32.52 | 27.51 | 27.58 | 34.01 | 28.24 | 26.00 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 52.79 | 51.12 | 43.03 | 37.11 | 37.15 | 44.05 | 37.58 | 30.52 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 46.99 | 45.12 | 36.42 | 30.47 | 30.20 | 36.69 | 28.51 | 26.37 |

## high_school — 5-shot

Acurácia micro (%).

| Modelo | original | all21 | atleast19 | atleast17 | families80 | compact4 | compact3 | reference3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Polygl0t/Tucano2-qwen-0.5B-Instruct | 36.17 | 33.91 | 26.03 | 22.91 | 22.97 | 25.44 | 20.55 | 22.93 |
| Polygl0t/Tucano2-qwen-1.5B-Instruct | 43.14 | 41.13 | 32.32 | 26.83 | 26.93 | 33.88 | 28.27 | 23.97 |
| Polygl0t/Tucano2-qwen-3.7B-Instruct | 52.10 | 50.41 | 42.36 | 36.28 | 36.58 | 43.67 | 37.52 | 27.14 |
| Qwen/Qwen3.5-0.8B | 37.98 | 35.79 | 27.12 | 24.04 | 24.59 | 26.91 | 21.28 | 23.97 |
| Qwen/Qwen3.5-27B | 69.54 | 68.47 | 63.14 | 58.75 | 58.97 | 63.66 | 58.99 | 42.17 |
| Qwen/Qwen3.5-2B | 46.14 | 44.24 | 35.49 | 30.01 | 30.56 | 36.71 | 30.20 | 27.25 |
| Qwen/Qwen3.5-35B-A3B | 65.22 | 63.99 | 57.93 | 53.05 | 53.35 | 58.63 | 53.30 | 38.63 |
| Qwen/Qwen3.5-4B | 57.48 | 55.98 | 48.69 | 42.98 | 43.38 | 49.54 | 43.46 | 32.64 |
| Qwen/Qwen3.5-9B | 59.56 | 58.13 | 51.11 | 45.50 | 45.92 | 51.95 | 46.07 | 32.00 |
| gemini-2.5-flash | 72.91 | 71.95 | 67.22 | 63.34 | 63.51 | 67.76 | 63.75 | 50.65 |
| gemini-3.8-flash | 89.73 | 89.37 | 87.61 | 86.16 | 86.23 | 87.86 | 86.46 | 80.50 |
| google/gemma-4-12B-it | 61.52 | 60.16 | 53.57 | 48.52 | 48.95 | 54.53 | 49.66 | 33.94 |
| google/gemma-4-26B-A4B-it | 64.80 | 63.55 | 57.51 | 52.87 | 53.17 | 58.41 | 54.02 | 37.83 |
| google/gemma-4-31B-it | 69.32 | 68.24 | 62.99 | 58.83 | 59.16 | 63.76 | 59.91 | 41.75 |
| google/gemma-4-E2B-it | 46.75 | 44.87 | 35.95 | 30.02 | 30.63 | 36.35 | 28.24 | 25.57 |
| google/gemma-4-E4B-it | 56.27 | 54.73 | 47.16 | 41.12 | 41.46 | 48.00 | 41.71 | 29.80 |
| meta-llama/Llama-3.1-8B-Instruct | 48.74 | 46.93 | 38.34 | 32.06 | 32.38 | 39.79 | 33.40 | 25.68 |
| meta-llama/Llama-3.2-1B-Instruct | 31.92 | 29.51 | 24.00 | 22.72 | 22.56 | 22.93 | 19.27 | 23.87 |
| meta-llama/Llama-3.2-3B-Instruct | 42.38 | 40.35 | 31.25 | 25.97 | 26.03 | 32.60 | 26.57 | 24.91 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 53.41 | 51.77 | 43.81 | 37.79 | 37.80 | 44.86 | 38.91 | 30.45 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 46.82 | 44.94 | 36.13 | 30.36 | 30.01 | 36.40 | 28.27 | 26.13 |

## undergraduate — média 0/5-shot

Acurácia micro (%).

| Modelo | original | all21 | atleast19 | atleast17 | families80 | compact4 | compact3 | reference3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Polygl0t/Tucano2-qwen-0.5B-Instruct | 33.39 | 32.23 | 26.74 | 23.33 | 23.49 | 24.44 | 18.66 | 24.10 |
| Polygl0t/Tucano2-qwen-1.5B-Instruct | 37.35 | 36.25 | 30.79 | 26.29 | 26.50 | 30.92 | 26.25 | 25.83 |
| Polygl0t/Tucano2-qwen-3.7B-Instruct | 48.68 | 47.78 | 42.54 | 36.70 | 37.06 | 42.84 | 37.45 | 28.69 |
| Qwen/Qwen3.5-0.8B | 37.66 | 36.57 | 30.09 | 25.24 | 26.27 | 27.35 | 18.73 | 26.68 |
| Qwen/Qwen3.5-27B | 76.06 | 75.64 | 72.58 | 68.27 | 68.77 | 72.57 | 68.81 | 34.81 |
| Qwen/Qwen3.5-2B | 46.55 | 45.62 | 39.24 | 32.20 | 33.37 | 38.73 | 31.52 | 28.01 |
| Qwen/Qwen3.5-35B-A3B | 72.96 | 72.49 | 69.05 | 64.23 | 64.81 | 68.98 | 64.66 | 39.65 |
| Qwen/Qwen3.5-4B | 60.95 | 60.26 | 55.40 | 48.89 | 49.74 | 55.30 | 49.68 | 31.96 |
| Qwen/Qwen3.5-9B | 65.77 | 65.17 | 60.86 | 54.95 | 55.71 | 60.89 | 55.64 | 33.65 |
| gemini-2.5-flash | 80.38 | 80.04 | 77.54 | 74.12 | 74.41 | 77.63 | 74.82 | 53.27 |
| gemini-3.8-flash | 92.52 | 92.39 | 91.47 | 90.31 | 90.36 | 91.73 | 90.82 | 79.64 |
| google/gemma-4-12B-it | 69.93 | 69.40 | 65.60 | 60.48 | 61.10 | 65.62 | 61.11 | 36.34 |
| google/gemma-4-26B-A4B-it | 76.47 | 76.06 | 73.10 | 69.06 | 69.55 | 73.26 | 69.86 | 45.33 |
| google/gemma-4-31B-it | 79.99 | 79.64 | 77.11 | 73.58 | 74.01 | 77.17 | 74.28 | 45.52 |
| google/gemma-4-E2B-it | 50.76 | 49.90 | 43.78 | 36.06 | 37.14 | 42.02 | 32.72 | 28.17 |
| google/gemma-4-E4B-it | 61.79 | 61.12 | 56.27 | 49.65 | 50.44 | 56.01 | 49.87 | 31.28 |
| meta-llama/Llama-3.1-8B-Instruct | 51.79 | 50.95 | 45.06 | 37.65 | 38.24 | 44.76 | 37.95 | 29.53 |
| meta-llama/Llama-3.2-1B-Instruct | 31.85 | 30.66 | 26.22 | 23.63 | 23.78 | 23.24 | 18.22 | 25.54 |
| meta-llama/Llama-3.2-3B-Instruct | 42.98 | 41.98 | 35.57 | 28.98 | 29.31 | 35.20 | 28.43 | 27.89 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 55.58 | 54.80 | 49.27 | 41.83 | 42.31 | 48.71 | 41.87 | 31.10 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 46.94 | 46.01 | 39.79 | 32.59 | 32.37 | 37.82 | 28.59 | 28.36 |

## undergraduate — 0-shot

Acurácia micro (%).

| Modelo | original | all21 | atleast19 | atleast17 | families80 | compact4 | compact3 | reference3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Polygl0t/Tucano2-qwen-0.5B-Instruct | 35.11 | 33.98 | 28.00 | 23.89 | 24.13 | 25.34 | 18.19 | 23.91 |
| Polygl0t/Tucano2-qwen-1.5B-Instruct | 38.36 | 37.28 | 32.07 | 27.13 | 27.45 | 31.97 | 26.84 | 25.66 |
| Polygl0t/Tucano2-qwen-3.7B-Instruct | 47.24 | 46.32 | 41.39 | 35.71 | 36.10 | 41.69 | 36.38 | 27.33 |
| Qwen/Qwen3.5-0.8B | 37.39 | 36.29 | 30.06 | 25.44 | 26.46 | 27.37 | 19.21 | 26.85 |
| Qwen/Qwen3.5-27B | 75.84 | 75.42 | 72.34 | 68.00 | 68.50 | 72.31 | 68.58 | 34.22 |
| Qwen/Qwen3.5-2B | 46.57 | 45.64 | 39.20 | 32.00 | 33.20 | 38.59 | 31.09 | 28.05 |
| Qwen/Qwen3.5-35B-A3B | 72.92 | 72.45 | 69.00 | 64.16 | 64.73 | 68.89 | 64.51 | 39.56 |
| Qwen/Qwen3.5-4B | 60.77 | 60.09 | 55.20 | 48.68 | 49.57 | 55.13 | 49.56 | 32.16 |
| Qwen/Qwen3.5-9B | 65.95 | 65.35 | 61.08 | 55.15 | 55.94 | 61.10 | 55.84 | 34.58 |
| gemini-2.5-flash | 78.31 | 77.93 | 75.19 | 71.39 | 71.73 | 75.22 | 72.01 | 49.25 |
| gemini-3.8-flash | 92.87 | 92.74 | 91.86 | 90.74 | 90.80 | 92.11 | 91.21 | 80.58 |
| google/gemma-4-12B-it | 68.42 | 67.87 | 63.88 | 58.56 | 59.22 | 63.84 | 59.07 | 34.11 |
| google/gemma-4-26B-A4B-it | 75.66 | 75.24 | 72.16 | 67.93 | 68.45 | 72.28 | 68.75 | 43.24 |
| google/gemma-4-31B-it | 78.83 | 78.46 | 75.77 | 71.98 | 72.45 | 75.77 | 72.71 | 42.34 |
| google/gemma-4-E2B-it | 50.45 | 49.58 | 43.39 | 35.64 | 36.75 | 41.61 | 32.34 | 26.87 |
| google/gemma-4-E4B-it | 61.40 | 60.72 | 55.82 | 49.13 | 49.94 | 55.57 | 49.44 | 30.04 |
| meta-llama/Llama-3.1-8B-Instruct | 51.23 | 50.38 | 44.45 | 37.05 | 37.65 | 44.20 | 37.51 | 29.46 |
| meta-llama/Llama-3.2-1B-Instruct | 32.37 | 31.19 | 26.62 | 23.86 | 23.96 | 23.45 | 18.14 | 25.60 |
| meta-llama/Llama-3.2-3B-Instruct | 43.16 | 42.16 | 36.05 | 29.62 | 29.99 | 35.89 | 29.71 | 28.38 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 54.97 | 54.18 | 48.57 | 40.97 | 41.50 | 47.90 | 40.76 | 31.18 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 47.39 | 46.47 | 40.25 | 32.95 | 32.76 | 38.29 | 28.95 | 28.44 |

## undergraduate — 5-shot

Acurácia micro (%).

| Modelo | original | all21 | atleast19 | atleast17 | families80 | compact4 | compact3 | reference3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Polygl0t/Tucano2-qwen-0.5B-Instruct | 31.67 | 30.48 | 25.48 | 22.77 | 22.85 | 23.53 | 19.13 | 24.29 |
| Polygl0t/Tucano2-qwen-1.5B-Instruct | 36.33 | 35.22 | 29.51 | 25.45 | 25.55 | 29.88 | 25.67 | 26.00 |
| Polygl0t/Tucano2-qwen-3.7B-Instruct | 50.12 | 49.24 | 43.69 | 37.70 | 38.03 | 43.98 | 38.53 | 30.04 |
| Qwen/Qwen3.5-0.8B | 37.93 | 36.85 | 30.12 | 25.04 | 26.09 | 27.33 | 18.26 | 26.51 |
| Qwen/Qwen3.5-27B | 76.28 | 75.86 | 72.83 | 68.55 | 69.05 | 72.83 | 69.05 | 35.40 |
| Qwen/Qwen3.5-2B | 46.54 | 45.60 | 39.27 | 32.40 | 33.53 | 38.87 | 31.95 | 27.98 |
| Qwen/Qwen3.5-35B-A3B | 73.00 | 72.53 | 69.11 | 64.31 | 64.89 | 69.08 | 64.82 | 39.73 |
| Qwen/Qwen3.5-4B | 61.12 | 60.44 | 55.59 | 49.11 | 49.91 | 55.47 | 49.80 | 31.76 |
| Qwen/Qwen3.5-9B | 65.58 | 64.98 | 60.65 | 54.75 | 55.48 | 60.67 | 55.45 | 32.73 |
| gemini-2.5-flash | 82.45 | 82.14 | 79.90 | 76.84 | 77.10 | 80.04 | 77.63 | 57.28 |
| gemini-3.8-flash | 92.18 | 92.04 | 91.08 | 89.87 | 89.91 | 91.35 | 90.42 | 78.70 |
| google/gemma-4-12B-it | 71.43 | 70.93 | 67.32 | 62.41 | 62.98 | 67.39 | 63.15 | 38.58 |
| google/gemma-4-26B-A4B-it | 77.27 | 76.88 | 74.05 | 70.19 | 70.65 | 74.25 | 70.97 | 47.42 |
| google/gemma-4-31B-it | 81.16 | 80.83 | 78.44 | 75.18 | 75.57 | 78.58 | 75.86 | 48.69 |
| google/gemma-4-E2B-it | 51.07 | 50.21 | 44.16 | 36.49 | 37.54 | 42.43 | 33.10 | 29.46 |
| google/gemma-4-E4B-it | 62.18 | 61.52 | 56.72 | 50.17 | 50.94 | 56.44 | 50.30 | 32.53 |
| meta-llama/Llama-3.1-8B-Instruct | 52.36 | 51.52 | 45.66 | 38.25 | 38.84 | 45.32 | 38.40 | 29.61 |
| meta-llama/Llama-3.2-1B-Instruct | 31.32 | 30.12 | 25.82 | 23.40 | 23.61 | 23.03 | 18.30 | 25.49 |
| meta-llama/Llama-3.2-3B-Instruct | 42.81 | 41.81 | 35.08 | 28.35 | 28.64 | 34.52 | 27.15 | 27.41 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 56.19 | 55.42 | 49.97 | 42.69 | 43.13 | 49.52 | 42.98 | 31.02 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 46.49 | 45.56 | 39.33 | 32.23 | 31.97 | 37.36 | 28.22 | 28.28 |

## Reprodução e interpretação

Executar na raiz do projeto:

```bash
python ablations/hard/analyze_5shot.py
python ablations/hard/analyze_mean.py
python ablations/hard/update_tables.py
```

`counts.csv` registra as contagens por nível; `scores.csv` contém N, scores por protocolo, média, papel na seleção e IC de 95% da média.
`provenance.json` registra painéis, limiares e hashes das análises, predições, snapshot e tabelas.
ICs da média usam bootstrap pareado por questão (2.000 reamostragens, seed 42). São condicionais ao subconjunto fixo e não corrigem viés de seleção ou dependência entre questões da mesma prova.
As regras são exploratórias; escolher um limiar após observar scores torna a decisão adaptativa. Os scores dos selecionadores não são uma medida independente de dificuldade.
A metodologia principal está em [docs/hard_direct_methodology.md](../../docs/hard_direct_methodology.md).
