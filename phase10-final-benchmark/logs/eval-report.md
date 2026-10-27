## Leaderboard (fully right, 95% interval)

| model | backend | A reasoning | B extraction | C tool choice | D episodes | E long context |
|---|---|---|---|---|---|---|
| bonsai-2:27b | ollama | 93% (89%-96%) | 99% (96%-100%) | 100% (98%-100%) | 98% (91%-100%) | 100% (92%-100%) |
| gemma4:e4b | ollama | 80% (74%-85%) | 100% (98%-100%) | 96% (92%-98%) | 82% (70%-89%) | 100% (92%-100%) |
| qwen3.5:4b-bf16 | ollama | 92% (87%-95%) | 47% (40%-55%) | 97% (92%-99%) | 82% (70%-89%) | 100% (92%-100%) |
| nemotron-3.5-bolt:e4b | ollama | 80% (73%-85%) | 99% (95%-100%) | 89% (83%-93%) | 72% (59%-81%) | 100% (92%-100%) |
| gpt-oss:20b | ollama | 74% (68%-80%) | 99% (96%-100%) | 80% (73%-86%) | 62% (49%-73%) | 98% (88%-100%) |
| qwen3:14b | ollama | 68% (62%-75%) | 79% (71%-84%) | 89% (83%-93%) | 62% (49%-73%) | 98% (88%-100%) |
| bonsai-2:9b | ollama | 87% (82%-91%) | 99% (96%-100%) | 91% (86%-95%) | 60% (47%-71%) | 98% (88%-100%) |
| gemma4:12b | ollama | 82% (77%-87%) | 100% (98%-100%) | 92% (87%-95%) | 60% (47%-71%) | 100% (92%-100%) |
| ministral-3:14b | ollama | 73% (66%-79%) | 100% (98%-100%) | 39% (32%-47%) | 37% (26%-49%) | 98% (88%-100%) |
| qwen2.5:14b | ollama | 80% (74%-85%) | 99% (96%-100%) | 85% (79%-90%) | 25% (16%-37%) | 100% (92%-100%) |
| qwen2.5:7b-instruct-q8_0 | ollama | 62% (56%-69%) | 69% (61%-76%) | 85% (78%-90%) | 18% (11%-30%) | 100% (92%-100%) |
| qwen2.5:3b | ollama | 58% (52%-65%) | 39% (31%-47%) | 84% (77%-89%) | 12% (6%-22%) | 98% (88%-100%) |
| qwen2.5:7b | ollama | 60% (53%-67%) | 69% (62%-76%) | 91% (85%-94%) | 8% (4%-18%) | 98% (88%-100%) |
| qwen2.5:3b-instruct-q8_0 | ollama | 64% (57%-70%) | 54% (46%-62%) | 85% (78%-90%) | 2% (0%-9%) | 100% (92%-100%) |
| gemma3:12b | ollama | 76% (69%-81%) | 97% (93%-99%) | - | - | 80% (66%-89%) |
| llama3.1:8b | ollama | 51% (44%-58%) | 99% (95%-100%) | 76% (69%-82%) | 0% (0%-6%) | 60% (45%-73%) |
| qwen2.5-1.5b-instruct | vllm | 44% (37%-50%) | 67% (59%-74%) | 73% (65%-79%) | 0% (0%-6%) | 76% (61%-86%) |
| qwen2.5-3b-instruct | vllm | 62% (55%-68%) | 51% (43%-59%) | 87% (81%-92%) | 0% (0%-6%) | 96% (85%-99%) |
| qwen2.5:1.5b | ollama | 37% (31%-44%) | 77% (69%-83%) | 71% (64%-78%) | 0% (0%-6%) | 93% (82%-98%) |

## Suite D against the leader (exact McNemar on the 60 episodes)

| model | D pooled (both runs) | 95% interval | flips between runs | leader only | model only | p |
|---|---|---|---|---|---|---|
| bonsai-2:27b | 98% (118/120) | 94%-100% | 0 | - | - | - |
| qwen3.5:4b-bf16 | 82% (98/120) | 74%-88% | 0 | 10 | 0 | 0.00195 |
| gemma4:e4b | 81% (97/120) | 73%-87% | 1 | 10 | 0 | 0.00195 |
| nemotron-3.5-bolt:e4b | 72% (86/120) | 63%-79% | 0 | 16 | 0 | 3.05e-05 |
| bonsai-2:9b | 60% (72/120) | 51%-68% | 0 | 23 | 0 | 2.38e-07 |
| gemma4:12b | 60% (72/120) | 51%-68% | 0 | 23 | 0 | 2.38e-07 |
| qwen3:14b | 60% (72/120) | 51%-68% | 2 | 22 | 0 | 4.77e-07 |
| gpt-oss:20b | 59% (71/120) | 50%-68% | 9 | 22 | 0 | 4.77e-07 |
| qwen2.5:14b | 24% (29/120) | 17%-33% | 7 | 44 | 0 | 1.14e-13 |
| qwen2.5:1.5b | 0% (0/120) | 0%-3% | 0 | 59 | 0 | 3.47e-18 |

## Cost and the Pareto front (one episode at a time)

Pass rate: suite D pooled over both runs (120) for Ollama defaults, the 60 quality-run episodes for vLLM, the 20 Stage 2 episodes for spec-0.5b.

| model | config | D pass | n | s/episode | s/turn | VRAM GB | correct/hour | episodes/min | at once (episodes/min) | front |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:e4b | default | 81% | 120 | 3.5 | 0.29 | 5.4 | 839 | 17.3 | np4: 21.6 | yes |
| nemotron-3.5-bolt:e4b | default | 72% | 120 | 4.9 | 0.31 | 9.1 | 528 | 12.3 | np4: 11.5, np16: 5.8 |  |
| bonsai-2:9b | default | 60% | 120 | 7.1 | 0.45 | 3.9 | 305 | 8.5 | np4: 6.3 queued | yes |
| gpt-oss:20b | default | 59% | 120 | 7.1 | 0.46 | 12.4 | 301 | 8.5 | np4: no fit, np16: no fit |  |
| qwen3.5:4b-bf16 | default | 82% | 120 | 12.5 | 0.89 | 9.6 | 236 | 4.8 | np4: 4.5 queued, np16: 4.3 queued | yes |
| qwen3:14b | default | 60% | 120 | 11.1 | 0.70 | 10.1 | 194 | 5.4 | np4: no fit, np16: no fit |  |
| bonsai-2:27b | default | 98% | 120 | 19.3 | 1.21 | 9.1 | 184 | 3.1 | np4: 2.2 queued | yes |
| gemma4:12b | default | 60% | 120 | 11.9 | 0.77 | 8.6 | 181 | 5.0 | np4: 8.5, np16: no fit |  |
| qwen2.5:14b | default | 24% | 120 | 11.0 | 1.49 | 10.0 | 79 | 5.4 | np4: no fit, np16: no fit |  |
| qwen2.5:14b | spec-0.5b | 10% | 20 | 6.7 | 1.13 | 10.4 | 53 | 8.9 | - |  |
| qwen2.5:1.5b | vllm-tuned | 0% | 60 | 0.5 | 0.24 | WSL | 0 | 110.1 | vllm-c4: 372.6, vllm-c16: 895.9 |  |

wrote H:\iwow\Development\llm-inference-optimization-lab\phase10-final-benchmark\logs\eval-pareto.svg

## Warnings for evaluated models

**gemma4:e4b** (detect, preflight, quick, quality, repeat, cost)
- detect: Modelfile bundles a draft model (Ollama speculative decoding); nemotron's gained nothing alone and blocked batching
- detect: loads a vision/audio projector, which takes VRAM the agent does not use
- cost: 1 of 8 episodes change answer at NUM_PARALLEL 4 (batched decode; gemma4 broke this way)

**bonsai-2:27b** (detect, preflight, quick, quality, repeat, cost)
- detect: loads a vision projector, which takes VRAM the agent does not use
- cost: Ollama forces -np 1 for architecture qwen35: NUM_PARALLEL only queues (like qwen3.5)
- cost: fan-out buys little: 2.2 episodes/min at np4 vs 3.1 one at a time

**bonsai-2:9b** (detect, preflight, quick, quality, repeat, cost)
- detect: Modelfile sets {'presence_penalty': '1'}; pinned to neutral here, so the agent must pin it too or it runs a different model
- detect: loads a vision projector, which takes VRAM the agent does not use
- cost: Ollama forces -np 1 for architecture qwen35: NUM_PARALLEL only queues (like qwen3.5)
- cost: fan-out buys little: 6.3 episodes/min at np4 vs 8.5 one at a time

