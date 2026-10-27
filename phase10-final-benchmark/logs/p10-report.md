## Leaderboard (fully right, 95% interval)

| model | backend | A reasoning | B extraction | C tool choice | D episodes | E long context |
|---|---|---|---|---|---|---|
| qwen3.5:4b-bf16 | ollama | 92% (87%-95%) | 47% (40%-55%) | 97% (92%-99%) | 82% (70%-89%) | 100% (92%-100%) |
| nemotron-3.5-bolt:e4b | ollama | 80% (73%-85%) | 99% (95%-100%) | 89% (83%-93%) | 72% (59%-81%) | 100% (92%-100%) |
| gpt-oss:20b | ollama | 74% (68%-80%) | 99% (96%-100%) | 80% (73%-86%) | 62% (49%-73%) | 98% (88%-100%) |
| qwen3:14b | ollama | 68% (62%-75%) | 79% (71%-84%) | 89% (83%-93%) | 62% (49%-73%) | 98% (88%-100%) |
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
| qwen3.5:4b-bf16 | 82% (98/120) | 74%-88% | 0 | - | - | - |
| nemotron-3.5-bolt:e4b | 72% (86/120) | 63%-79% | 0 | 10 | 4 | 0.18 |
| gemma4:12b | 60% (72/120) | 51%-68% | 0 | 14 | 1 | 0.000977 |
| qwen3:14b | 60% (72/120) | 51%-68% | 2 | 14 | 2 | 0.00418 |
| gpt-oss:20b | 59% (71/120) | 50%-68% | 9 | 14 | 2 | 0.00418 |
| qwen2.5:14b | 24% (29/120) | 17%-33% | 7 | 34 | 0 | 1.16e-10 |
| qwen2.5:1.5b | 0% (0/120) | 0%-3% | 0 | 49 | 0 | 3.55e-15 |

## Phase 6's question: qwen2.5 7B Q4_K_M vs 3B Q8_0

| suite | 7B Q4_K_M | 3B Q8_0 | 7B only | 3B only | p |
|---|---|---|---|---|---|
| A | 60% | 64% | 15 | 22 | 0.324 |
| B | 69% | 54% | 49 | 26 | 0.0106 |
| C | 91% | 85% | 15 | 6 | 0.0784 |
| D | 8% | 2% | 4 | 0 | 0.125 |
| E | 98% | 100% | 0 | 1 | 1 |

## Cost and the Pareto front (one episode at a time)

Pass rate: suite D pooled over both runs (120) for Ollama defaults, the 60 quality-run episodes for vLLM, the 20 Stage 2 episodes for spec-0.5b.

| model | config | D pass | n | s/episode | s/turn | VRAM GB | correct/hour | episodes/min | at once (episodes/min) | front |
|---|---|---|---|---|---|---|---|---|---|---|
| nemotron-3.5-bolt:e4b | default | 72% | 120 | 4.9 | 0.31 | 9.1 | 528 | 12.3 | np4: 11.5, np16: 5.8 | yes |
| gpt-oss:20b | default | 59% | 120 | 7.1 | 0.46 | 12.4 | 301 | 8.5 | np4: no fit, np16: no fit |  |
| qwen3.5:4b-bf16 | default | 82% | 120 | 12.5 | 0.89 | 9.6 | 236 | 4.8 | np4: 4.5 queued, np16: 4.3 queued | yes |
| qwen3:14b | default | 60% | 120 | 11.1 | 0.70 | 10.1 | 194 | 5.4 | np4: no fit, np16: no fit |  |
| gemma4:12b | default | 60% | 120 | 11.9 | 0.77 | 8.6 | 181 | 5.0 | np4: 8.5, np16: no fit | yes |
| qwen2.5:14b | default | 24% | 120 | 11.0 | 1.49 | 10.0 | 79 | 5.4 | np4: no fit, np16: no fit |  |
| qwen2.5:14b | spec-0.5b | 10% | 20 | 6.7 | 1.13 | 10.4 | 53 | 8.9 | - |  |
| qwen2.5:1.5b | vllm-tuned | 0% | 60 | 0.5 | 0.24 | WSL | 0 | 110.1 | vllm-c4: 372.6, vllm-c16: 895.9 |  |

wrote H:\iwow\Development\llm-inference-optimization-lab\docs\phase10-pareto.svg
