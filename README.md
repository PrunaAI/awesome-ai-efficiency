# 🌟 Awesome AI Efficiency 🌟

![Awesome](https://awesome.re/badge.svg) ![MIT License](https://img.shields.io/badge/license-MIT-brightgreen)

A curated list of resources dedicated to enhancing efficiency in AI systems. This repository covers a wide range of topics essential for optimizing AI models and processes, aiming to make AI faster, cheaper, smaller, and greener!

### Topics Summary 🎨

| Topic            | Description                                    | Topics                                          |
|-------------------|------------------------------------------------|-------------------------------------------------------|
| **Quantization**  | Reducing precision of AI models without loss  | ![Quantization](https://img.shields.io/badge/Quantization-purple) |
| **Pruning**       | Removing unnecessary model parameters for efficiency | ![Pruning](https://img.shields.io/badge/Pruning-purple) |
| **Caching**       | Storing computation results for faster reuse  | ![Caching](https://img.shields.io/badge/Caching-purple) |
| **Distillation**  | Transferring knowledge from a large model to a smaller one | ![Distillation](https://img.shields.io/badge/Distillation-purple) |
| **Factorization** | Breaking down complex models into simpler, efficient components | ![Factorization](https://img.shields.io/badge/Factorization-purple) |
| **Compilation**   | Optimizing model code for specific hardware and environments | ![Compilation](https://img.shields.io/badge/Compilation-purple) |
| **Parameter-Efficient Fine-tuning** | Learning a subset of parameters | ![PEFT](https://img.shields.io/badge/Peft-purple) |
| **Speculative Decoding** | Learning a subset of parameters | ![SpecDec](https://img.shields.io/badge/SpecDec-purple) |
| **Hardware**      | Leveraging specialized hardware for faster model execution | ![Hardware](https://img.shields.io/badge/Hardware-purple) |
| **Training**      | Techniques for making model training faster and more efficient | ![Training](https://img.shields.io/badge/Training-purple) |
| **Inference**     | Optimizing the speed and resource usage during model inference | ![Inference](https://img.shields.io/badge/Inference-purple) |
| **Sustainability** | Strategies to reduce the environmental impact of AI systems | ![Sustainability](https://img.shields.io/badge/Sustainability-purple) |
| **Scalability**   | Approaches for scaling AI models and infrastructure efficiently | ![Scalability](https://img.shields.io/badge/Scalability-purple) |

If you find this list helpful, give it a ⭐ on GitHub, share it, and contribute by submitting a pull request or issue!

---

## Table of Contents
- [Facts/Numbers 📊](#factsnumbers-📊)
- [Tools 🛠️](#tools-🛠️)
- [Articles 📰](#articles-📰)
- [Research Papers 📄](#research-papers-📄)
- [Books 📚](#books-📚)
- [Lectures 🎓](#lectures-🎓)
- [People 🧑‍💻](#people-🧑‍💻)
- [Organizations 🌍](#Organizations-🌍)

---

## Facts 📊
- **2 nuclear plants**: Number of nuclear plants to constantly work ot generate enough energy if 80M people generate 5 pages per day ([Source](https://huggingface.co/spaces/genai-impact/ecologits-calculator), 2025)
- **1 smartphone charge**: Amount of energy required to AI generate a couple of images or run a few thousands inference with an LLM ([Source](https://arxiv.org/pdf/2311.16863), 2024)
- **>10s**: Time requried to generate 1 HD image with Flux on H100 or to generate 100 tokens with Llama 3 on T4 ([Source](https://flux-pruna-benchmark.vercel.app/) and [Source](https://huggingface.co/spaces/optimum/llm-perf-leaderboard), 2024)
- **61,848.0x**: Difference between the highest and lowest energy use in energy leaderboard for AI models ([Source](https://huggingface.co/spaces/AIEnergyScore/Leaderboard), 2025).
- **1,300MWh**: GPT-3, for example, is estimated to use just under 1,300 megawatt hours (MWh) of electricity; about as much power as consumed annually by 130 US homes ([Source](https://www.theverge.com/24066646/ai-electricity-energy-watts-generative-consumption))

---

## Tools 🛠️
- :heart: **[Pruna](https://docs.pruna.ai/en/latest/)** :heart:: A package to make AI models faster, smaller, faster, greener by combining compression methods (incl. quantization, pruning, caching, compilation, distillation...) on various hardware.
- **[TensorRT](https://developer.nvidia.com/tensorrt)**: High-performance deep learning inference library for NVIDIA GPUs.
- **[ONNX](https://onnx.ai/)**: Open Neural Network Exchange format for interoperability among deep learning frameworks.
- **[Code Carbon](https://mlco2.github.io/codecarbon/)**: Library to track energy and carbon efficiency of various hardware.
- **[LLM Perf](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)**: A framework for benchmarking the performance of transformers models with different hardwares, backends and optimizations.
- **[AI Energy Score](https://huggingface.co/spaces/AIEnergyScore/submission_portal)**: An initiative to establish comparable energy efficiency ratings for AI models, helping the industry make informed decisions about sustainability in AI development.
- **[Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)**: TensorFlow toolkit for optimizing machine learning models for deployment and execution.
- **[Green Coding](https://green-coding.ai/)**: LLM service that you can use to prompt most open source models and see the resource usage.
- **[EcoLogits](https://huggingface.co/spaces/genai-impact/ecologits-calculator)**: EcoLogits is a python library that tracks the energy consumption and environmental footprint of using generative AI models through APIs.
- **[Fast Tokenizer](https://github.com/NLPOptimize/flash-tokenizer)**: Fast tokenizer is an efficient and optimized tokenizer engine for llm inference serving.

---

## Articles 📰
- *"[What's the environmental cost of AI?](https://www.co2ai.com/insights/whats-the-environmental-cost-of-ai)"* (2024) - CO2 AI
- *"[Shrinking the giants: Paving the way for TinyAI](https://www.cell.com/device/abstract/S2666-9986(24)00247-3)"* (2024) - Cell Press
- *"[DeepSeek might not be such good news for energy after all](https://www.technologyreview.com/2025/01/31/1110776/deepseek-might-not-be-such-good-news-for-energy-after-all/)"* (2024) - MIT Technology Review
- *"[AI already uses as much energy as a small country. It’s only the beginning.](https://www.vox.com/climate/2024/3/28/24111721/climate-ai-tech-energy-demand-rising)"* (2024) - Vox
- *"[Quelle contribution du numérique à la décarbonation ?](https://www.strategie.gouv.fr/publications/contribution-numerique-decarbonation)"* (2024) - France Stratégie
- *"[Les promesses de l’IA grevées par un lourd bilan carbone](https://www.lemonde.fr/planete/article/2024/08/04/climat-les-promesses-de-l-ia-grevees-par-un-lourd-bilan-carbone_6266586_3244.html)"* (2024) - Le Monde
- *"[How much electricity does AI consume?](https://www.theverge.com/24066646/ai-electricity-energy-watts-generative-consumption)"* (2024) - The Verge
- *"[How do I track the direct environmental impact of my own inference and training when working with AI?](https://rtl.chrisadams.me.uk/2024/08/how-do-i-track-my-direct-environmental-impact-of-my-own-inference-and-training-when-working-with-ai/)"* (2024) - Blog
- *"[Data center emissions probably 662% higher than big tech claims. Can it keep up the ruse?](https://www.theguardian.com/technology/2024/sep/15/data-center-gas-emissions-tech)"* (2024) - The Guardian
- *"[Light bulbs have energy ratings — so why can’t AI chatbots?](https://www.nature.com/articles/d41586-024-02680-3)"* (2024) - Nature
- *"[The Environmental Impacts of AI -- Primer](https://huggingface.co/blog/sasha/ai-environment-primer
)"* (2024) - Hugging Face
- *"[The Climate and Sustainability Implications of Generative AI](https://mit-genai.pubpub.org/pub/8ulgrckc/release/2)"* (2024) - MIT
- *"[AI's "eye-watering" use of resources could be a hurdle to achieving climate goals, argue experts](https://www.dezeen.com/2023/08/09/ai-resources-climate-environment-energy-aitopia/)"* (2023) - dezeen
- *"[How coders can help save the planet?](https://trellis.net/article/how-coders-can-help-save-planet/)"* (2023) - Blog
- *"[Reducing the Carbon Footprint of Generative AI](https://www.linkedin.com/pulse/reducing-carbon-footprint-generative-ai-boris-gamazaychikov/)"* (2023) - Blog
- *"[The MPG of LLMs: Exploring the Energy Efficiency of Generative AI](https://www.linkedin.com/pulse/mpg-llms-exploring-energy-efficiency-generative-ai-gamazaychikov/)"* (2023) - Blog
- *"[Ecologie numérique: L’IA durable, entre vœu pieux et opportunité de marché](https://www.liberation.fr/economie/economie-numerique/lia-durable-entre-voeu-pieux-et-opportunite-de-marche-20250210_RRW3GZT5KFDSRCH4MIAYXKFKYE/?at_creation=Fil_Vert_2025-02-10&at_campaign=NL_FILVERT&at_email_type=acquisition&at_medium=email&actId=%7EaZ0rwLtGEnqhYxPXKI1iJ5Y1MdiymPA0SNCFqk5sBgGT7glf8EeY1JVPC1N7NQGN_CpyjQGeTBfhOG8JG34Nc4BinZSv7tVnM0VzpQUN77jrV4B7cXevAWfA%3D&actCampaignType=CAMPAIGN_MAIL&actSource=543696)"* (2025) - Libération

---

## Reports 📈
- *"[The 2025 AI Index Report](https://hai.stanford.edu/ai-index/2025-ai-index-report)"* (2025) - Stanford Human-centered Artificial Intelligence
- *"[Energy and AI](https://www.iea.org/reports/energy-and-ai)"* (2025) - International Energy Agency
- *"[Key challenges for the environmental performance of AI](https://www.sustainableaicoalition.org/key-challenges/)"* (2025) - French Ministry
- *"[Artificial Intelligence and electricity: A system dynamics approach](https://www.se.com/ww/en/insights/sustainability/sustainability-research-institute/artificial-intelligence-electricity-system-dynamics-approach/)"* (2024) - Schneider
- *"[Notable AI Models](https://epoch.ai/data/notable-ai-models)"* (2025) - Epoch AI
- *"[Powering artificial intelligence](Deloitte)"* (2024) - Deloitte
- *"[Google Sustainability Reports](https://sustainability.google/reports/)"* (2024) - Google
- *"[How much water does AI consume? The public deserves to know](https://oecd.ai/en/wonk/how-much-water-does-ai-consume)"* (2023) - OECD
- *"[Measuring the environmental impacts of artificial intelligence compute and applications](https://www.oecd.org/en/publications/measuring-the-environmental-impacts-of-artificial-intelligence-compute-and-applications_7babf571-en.html)"* (2022) - OECD

---

## Research Articles 📄
| Paper | Year | Venue | Tags |
|-------|------|-------|------|
| <sub><b>[BitNet b1.58 2B4T Technical Report](https://arxiv.org/pdf/2504.12285)</b></sub> | 2025 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[NdLinear Is All You Need for Representation Learning](https://arxiv.org/pdf/2503.17353)</b></sub> | 2025 | None | <sub><b>![Factorization](https://img.shields.io/badge/Factorization-purple)</b></sub> |
| <sub><b>[LoRI: Reducing Cross-Task Interference in Multi-Task LowRank Adaptation](https://arxiv.org/pdf/2504.07448)</b></sub> | 2025 | ICLR | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)</b></sub> |
| <sub><b>[FISH-Tuning: Enhancing PEFT Methods with Fisher Information](https://arxiv.org/pdf/2504.04050)</b></sub> | 2025 | None | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)</b></sub> |
| <sub><b>[Green Prompting](https://arxiv.org/pdf/2503.10666)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Compression Scaling Laws:Unifying Sparsity and Quantization](https://arxiv.org/abs/2502.16440)</b></sub> | 2025 | None | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality](https://arxiv.org/abs/2410.19355)</b></sub> | 2025 | ICLR | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[LANTERN: Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding](https://arxiv.org/abs/2410.03355)</b></sub> | 2025 | ICLR | <sub><b></b>![SpecDec](https://img.shields.io/badge/SpecDec-purple)</sub> |
| <sub><b>[Cache Me If You Must: Adaptive Key-Value Quantization for Large Language Models](https://arxiv.org/abs/2501.19392)</b></sub> | 2025 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Real-Time Video Generation with Pyramid Attention Broadcast](https://arxiv.org/abs/2408.12588)</b></sub> | 2025 | ICLR | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[Not All Prompts Are Made Equal: Prompt-based Pruning of Text-to-Image Diffusion Models](https://arxiv.org/abs/2406.12042)</b></sub> | 2025 | ICLR | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing](https://arxiv.org/abs/2502.15618)</b></sub> | 2025 | ICLR | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[FlexiDiT: Your Diffusion Transformer Can Easily Generate High-Quality Samples with Less Compute]()</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling](https://arxiv.org/pdf/2502.06703)</b></sub> | 2025 | None | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/pdf/2405.16406))</b></sub> | 2025 | ICLR | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Making AI Less “Thirsty”: Uncovering and Addressing the Secret Water Footprint of AI Models](https://arxiv.org/pdf/2304.03271))</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps](https://arxiv.org/pdf/2501.09732))</b></sub> | 2025 | None | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[Distillation Scaling Laws](https://arxiv.org/abs/2502.08606)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[QuEST: Stable Training of LLMs with 1-Bit Weights and Activations](https://arxiv.org/pdf/2502.05003))</b></sub> | 2025 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Distillation Scaling Laws](https://arxiv.org/pdf/2502.08606))</b></sub> | 2025 | None | <sub><b>![Distillation](https://img.shields.io/badge/Distillation-purple)</b></sub> |
| <sub><b>[From Efficiency Gains to Rebound Effects: The Problem of Jevons' Paradox in AI's Polarized Environmental Debate](https://arxiv.org/abs/2501.16548v1))</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/pdf/2404.00456)</b></sub> | 2024 | NeurIPS | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[The Iterative Optimal Brain Surgeon: Faster Sparse Recovery by Leveraging Second-Order Information](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fc1dce23e0ba01d3b691789cfd4f65c3-Abstract-Conference.html)</b></sub> | 2024 | NeurIPS | <sub><b></b></sub> |
| <sub><b>[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)</b></sub> | 2024 | MLSys | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[LOFIT: Localized Fine-tuning on LLM Representations](https://arxiv.org/pdf/2406.01563)</b></sub> | 2024 | NeurIPS | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)</b></sub> |
| <sub><b>[Outlier Weighed Layerwise Sparsity: A Missing Secret Sauce for Pruning LLMs to High Sparsity](https://arxiv.org/pdf/2310.05175)</b></sub> | 2024 | ICML | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality](https://arxiv.org/pdf/2410.19355)</b></sub> | 2024 | None | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396)</b></sub> | 2024 | ICML | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[QTIP: Quantization with Trellises and Incoherence Processing](https://arxiv.org/abs/2406.11235)</b></sub> | 2024 | NeurIPS | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2409.17066)</b></sub> | 2024 | EMNLP | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)</b></sub> | 2024 | NeurIPS | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532)</b></sub> | 2024 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/html/2401.06118v2)</b></sub> | 2024 | ICML | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Fast Matrix Multiplications for Lookup Table-Quantized LLMs](https://arxiv.org/html/2407.10960v1)</b></sub> | 2024 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[GPTVQ: The Blessing of Dimensionality for LLM Quantization](https://arxiv.org/html/2402.15319v1)</b></sub> | 2024 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey](https://arxiv.org/abs/2403.14608)</b></sub> | 2024 | None | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)</b></sub> |
| <sub><b>[SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration](https://arxiv.org/pdf/2410.06916)</b></sub> | 2024 | None | <sub><b>![SpecDec](https://img.shields.io/badge/SpecDec-purple)</b></sub> |
| <sub><b>[SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices](https://arxiv.org/pdf/2406.02532)</b></sub> | 2024 | NeurIPS | <sub><b>![SpecDec](https://img.shields.io/badge/SpecDec-purple)</b></sub> |
| <sub><b>[ShortGPT: Layers in Large Language Models are More Redundant Than You Expect]()</b>https://arxiv.org/pdf/2403.03853</sub> | 2024 | None | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Canvas: End-to-End Kernel Architecture Search in Neural Networks](https://arxiv.org/pdf/2304.07741)</b></sub> | 2024 | None | <sub><b>![Compilation](https://img.shields.io/badge/Compilation-purple)</b></sub> |
| <sub><b>[Scaling Laws for Precision](https://arxiv.org/pdf/2411.04330)</b></sub> | 2024 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[DeepCache: Accelerating Diffusion Models for Free](https://arxiv.org/pdf/2312.00858)</b></sub> | 2024 | CVPR | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding](https://arxiv.org/abs/2401.07851)</b></sub> | 2024 | ACL | <sub><b>![Distillation](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[Power Hungry Processing: Watts Driving the Cost of AI Deployment?](https://dl.acm.org/doi/pdf/10.1145/3630106.3658542)</b></sub> | 2024 | FaccT | <sub><b></b></sub> |
| <sub><b>[Decoding Compressed Trust: Scrutinizing the Trustworthiness of Efficient LLMs Under Compression](https://arxiv.org/pdf/2310.05175)</b></sub> | 2024 | ICML | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Pushing the Limits of Large Language Model Quantization via the Linearity Theorem](https://arxiv.org/abs/2411.17525)</b></sub> | 2024 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Position: Tensor Networks are a Valuable Asset for Green AI](https://arxiv.org/pdf/2205.12961)</b></sub> | 2024 | None | <sub><b>![Factorization](https://img.shields.io/badge/Factorization-purple)</b></sub> |
| <sub><b>[Hype, Sustainability, and the Price of the Bigger-is-Better Paradigm in AI](https://arxiv.org/pdf/2409.14160)</b></sub> | 2024 | None | <sub><b></b></sub> |
| <sub><b>[Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes](https://arxiv.org/abs/2402.05406)</b></sub> | 2024 | ICLR | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)</b></sub> | 2023 | SOSP | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[Broken Neural Scaling Laws](https://arxiv.org/abs/2210.14891)</b></sub> | 2023 | ICLR | <sub><b></b></sub> |
| <sub><b>[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)</b></sub> | 2023 | ICML | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs](https://arxiv.org/pdf/2312.15230)</b></sub> | 2023 | None | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Trends in AI inference energy consumption: Beyond the performance-vs-parameter laws of deep learning](https://www.sciencedirect.com/science/article/pii/S2210537923000124#:~:text=However%2C%20for%20deployed%20systems%2C%20inference,but%20inference%20is%20done%20repeatedly.)</b></sub> | 2023 | Sustainable Computing: Informatics and Systems | <sub><b></b></sub> |
| <sub><b>[An experimental comparison of software-based power meters: focus on CPU and GPU](https://inria.hal.science/hal-04030223v2/file/_CCGrid23__An_experimental_comparison_of_software_based_power_meters__from_CPU_to_GPU.pdf)</b></sub> | 2023 | CCGrid | <sub><b>![Hardware](https://img.shields.io/badge/Hardware-purple)</b></sub> |
| <sub><b>[Fast Inference from Transformers via Speculative Decoding](https://openreview.net/pdf?id=C9NEblP8vS)</b></sub> | 2023 | ICML | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)</b></sub> | 2023 | ICLR | <sub><b></b></sub> |
| <sub><b>[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/pdf/2210.17323)</b></sub> | 2023 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Knowledge Distillation: A Good Teacher is Patient and Consistent](https://arxiv.org/pdf/2106.05237))</b></sub> | 2022 | CVPR | <sub><b>![Distillation](https://img.shields.io/badge/Distillation-purple)</b></sub> |
| <sub><b>[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)</b></sub> | 2022 | ICLR | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)</b></sub> |
| <sub><b>[LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)</b></sub> | 2022 | NeurIPS | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Optimal Clipping and Magnitude-aware Differentiation for Improved Quantization-aware Training](https://arxiv.org/abs/2206.06501)</b></sub> | 2022 | ICML | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Learnable Lookup Table for Neural Network Quantization](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learnable_Lookup_Table_for_Neural_Network_Quantization_CVPR_2022_paper.pdf)</b></sub> | 2022 | CVPR | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)</b></sub> | 2022 | None | <sub><b></b></sub> |
| <sub><b>[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)</b></sub> | 2022 | None | <sub><b></b></sub> |
| <sub><b>[Towards a Unified View of Parameter-Efficient Transfer Learning](https://arxiv.org/abs/2110.04366)</b></sub> | 2022 | ICLR | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)</b></sub> |
| <sub><b>[Parameter-Efficient Transfer Learning with Diff Pruning](https://arxiv.org/abs/2012.07463)</b></sub> | 2021 | ACL | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033)</b></sub> | 2020 | MLSys | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Model Compression via Distillation and Quantization](https://arxiv.org/abs/1802.05668)</b></sub> | 2018 | ICLR | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Optimal Brain Damage](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)</b></sub> | 1989 | NeurIPs | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |

---

## Blogs 📰
- *"[Tensor Parallelism with CUDA - Multi-GPU Matrix Multiplication](https://substack.com/home/post/p-158663472)" (2025)* - Substack
- *"[Automating GPU Kernel Generation with DeepSeek-R1 and Inference Time Scaling](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/)" (2025)* - Nvidia Developer
- *"[AI CUDA Engineer](https://sakana.ai/ai-cuda-engineer/)" (2025)* - Sakana AI
- *"[The ML/AI Engineer's starter guide to GPU Programming](https://neuralbits.substack.com/p/the-mlai-engineers-starter-guide)" (2025)* - Neural Bits
- *"[Understanding Quantization for LLMs](https://medium.com/@lmpo/understanding-model-quantization-for-llms-1573490d44ad)" (2024)* - Medium
- *"[Don't Merge Your LoRA Adapter Into a 4-bit LLM](https://kaitchup.substack.com/p/dont-merge-your-lora-adapter-into?source=post_page-----2216ffcdc27b---------------------------------------)" (2023)* - Substack
- *"[Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-mem)" (2023)* - Nvidia Developer
- *"[GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#understand-perf)" (2023)* - Nvidia Developer

---

## Books 📚
- **[Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.com/dp/0323912311?ref_=cm_sw_r_cp_ud_dp_YVNSMFJMGQ9N457Z8Q6D)** (2022), Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj
- **[Efficient Deep Learning](https://efficientdlbook.com/)** (2022), Gaurav Menghani, Naresh Singh


---

## Lectures 🎓
- **[Data Compression, Theory and Applications]([https://www.youtube.com/c/MITHANLab](https://stanforddatacompressionclass.github.io/notes/contents.html#ee274-data-compression-course-notes))** (2024) - Stanford
- **[MIT Han's Lab](https://www.youtube.com/c/MITHANLab)** (2024) - MIT Lecture by Han's lab
- **[GPU Mode](https://www.youtube.com/@GPUMODE)** (2020) - Tutorials by GPU mode community

---

## People 🧑‍💻

| Name                  | Affiliation                       | Research Interests | Social Media |
|-----------------------|-----------------------------------|---------------------|--------------|
| **Saleh Ashkboos**        | ETH Zurich                        | <sub>Quantization</sub> | - |
| **Dan Alistar**        | IST Austria                        | <sub>AI Compression</sub> | - |
| **Elias Frantar**        | OpenAI                        | <sub>Quantization</sub> | - |
| **Tim Dettmers**        | CMU                        | <sub>Quantization</sub> | - |
| **Tim Dettmers**        | CMU                        | <sub>Quantization</sub> | - |
| **Song Han**        | MIT                         | <sub>AI Efficiency</sub> | - |
| **Scott Chamberlain**        | TBD                         | <sub>AI Efficiency</sub> | - |
| **Benoit Petit**        | Boavista                         | <sub>Data Center Efficiency</sub> | - |
| **Samuel Rincé**        | Gen AI Impact                                 | <sub>AI Efficiency, Sustainability</sub> | - |
| **Téo Alves Da Costa**           | Ekimetrics                       | <sub>AI Efficiency, Sustainability</sub> | - |
| **Sasha Luccioni**      | Hugging Face                       | <sub>AI Efficiency, Sustainability</sub> | - |

## Organizations 🌍

| Organization           | Description                                                              | Website                                              |
|------------------------|--------------------------------------------------------------------------|------------------------------------------------------|
| **Data4Good**           | A platform that connects data scientists with social impact projects to address global challenges using data. | [data4good.org](https://www.data4good.org/)           |
| **Make.org**            | A global platform that empowers citizens to propose and take action on social and environmental issues through collective projects. | [make.org](https://www.make.org/)                     |
| **CodeCarbon**          | A tool that helps track the carbon emissions of machine learning models and optimizes them for sustainability. | [codecarbon.io](https://www.codecarbon.io/)           |
| **Sustainable AI Coalition** | An organization dedicated to advancing sustainability in AI technologies and promoting best practices for green AI. | [sustainableaicoalition.org](https://www.sustainableaicoalition.org/) |

---

## Contributing 🤝
Contributions are welcome! Please follow our [contribution guidelines](CONTRIBUTING.md) to add new resources or suggest improvements that promote AI efficiency.

---

## License 📄
This project is licensed under the [MIT License](LICENSE). Feel free to share and use the resources as needed.
