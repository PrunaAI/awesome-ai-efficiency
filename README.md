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
| **Speculative Decoding** | Decoding with batches | ![SpecDec](https://img.shields.io/badge/SpecDec-purple) |
| **Hardware**      | Leveraging specialized hardware for faster model execution | ![Hardware](https://img.shields.io/badge/Hardware-purple) |
| **Training**      | Techniques for making model training faster and more efficient | ![Training](https://img.shields.io/badge/Training-purple) |
| **Inference**     | Optimizing the speed and resource usage during model inference | ![Inference](https://img.shields.io/badge/Inference-purple) |
| **Sustainability** | Strategies to reduce the environmental impact of AI systems | ![Sustainability](https://img.shields.io/badge/Sustainability-purple) |
| **Scalability**   | Approaches for scaling AI models and infrastructure efficiently | ![Scalability](https://img.shields.io/badge/Scalability-purple) |

If you find this list helpful, give it a ⭐ on GitHub, share it, and contribute by submitting a pull request or issue!

---

## Table of Contents

  - [Facts 📊](#facts-)
  - [Tools 🛠️](#tools-️)
  - [Articles 📰](#articles-)
  - [Reports 📈](#reports-)
  - [Research Articles 📄](#research-articles-)
  - [Blogs 📰](#blogs-)
  - [Books 📚](#books-)
  - [Lectures 🎓](#lectures-)
  - [People 🧑‍💻](#people-)
  - [Organizations 🌍](#organizations-)
  - [Contributing 🤝](#contributing-)
  - [License 📄](#license-)

---

## Facts 📊
- **3-40Wh**: Amount of energy consumed for one small to long ChatGPT query ([Source](https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use), 2025)  
- **1L**: Estimated amount of water required for 20-100 ChatGPT queries ([Source](https://arxiv.org/pdf/2304.03271), 2025)  
- **2 nuclear plants**: Number of nuclear plants to constantly work ot generate enough energy if 80M people generate 5 pages per day ([Source](https://huggingface.co/spaces/genai-impact/ecologits-calculator), 2025)
- **1 smartphone charge**: Amount of energy required to AI generate a couple of images or run a few thousands inference with an LLM ([Source](https://arxiv.org/pdf/2311.16863), 2024)
- **>10s**: Time requried to generate 1 HD image with Flux on H100 or to generate 100 tokens with Llama 3 on T4 ([Source](https://flux-pruna-benchmark.vercel.app/) and [Source](https://huggingface.co/spaces/optimum/llm-perf-leaderboard), 2024)
- **7-10 smartphone charges**: Amount of energy required to AI generate one video with Wan 2.1 ([Source](https://huggingface.co/blog/jdelavande/text-to-video-energy-cost))
- **61,848.0x**: Difference between the highest and lowest energy use in energy leaderboard for AI models ([Source](https://huggingface.co/spaces/AIEnergyScore/Leaderboard), 2025).
- **1,300MWh**: GPT-3, for example, is estimated to use just under 1,300 megawatt hours (MWh) of electricity; about as much power as consumed annually by 130 US homes ([Source](https://www.theverge.com/24066646/ai-electricity-energy-watts-generative-consumption), 2024)
- **800M users/week**: Amount of users using ChatGPT per week in 2025 ([Source](https://bettertech.blog/2025/04/19/ais-impacts-how-to-limit-them-and-why/))
- **1B messages/day**: Amount of ChatGPT queries per day in 2025 ([Source](https://bettertech.blog/2025/04/19/ais-impacts-how-to-limit-them-and-why/))
- **+160%**: Expected increase of data center power consumption by 2030 ([Source](https://bettertech.blog/2025/04/19/ais-impacts-how-to-limit-them-and-why/))
- **x3.8**: Hardware acceleration (GPU/TPU) reduces energy consumption by a factor of 3.8 compared with the CPU, for the same task, but also reduces response time by up to 39% ([Source](https://greenspector.com/en/the-environmental-impact-of-local-text-ai/))
- **x18**:The carbon footprint of a task can vary by a factor of 18 depending on the model, framework and backend used ([Source](https://greenspector.com/en/the-environmental-impact-of-local-text-ai/))

---

## Tools 🛠️
- :heart: **[Pruna](https://docs.pruna.ai/en/latest/)** :heart:: A package to make AI models faster, smaller, faster, greener by combining compression methods (incl. quantization, pruning, caching, compilation, distillation...) on various hardware.
- **[TensorRT](https://developer.nvidia.com/tensorrt)**: High-performance deep learning inference library for NVIDIA GPUs.
- **[ONNX](https://onnx.ai/)**: Open Neural Network Exchange format for interoperability among deep learning frameworks.
- **[Code Carbon](https://mlco2.github.io/codecarbon/)**: Library to track energy and carbon efficiency of various hardware.
- **[LLM Perf](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)**: A framework for benchmarking the performance of transformers models with different hardwares, backends and optimizations.
- **[ML.ENERGY Leaderboard](https://ml.energy/leaderboard/?__theme=dark?__theme=light)**: An initiative to benchmark energy efficiency of AI models.
- **[AI Energy Score](https://huggingface.co/spaces/AIEnergyScore/submission_portal)**: An initiative to establish comparable energy efficiency ratings for AI models, helping the industry make informed decisions about sustainability in AI development.
- **[Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)**: TensorFlow toolkit for optimizing machine learning models for deployment and execution.
- **[Green Coding](https://green-coding.ai/)**: LLM service that you can use to prompt most open source models and see the resource usage.
- **[EcoLogits](https://huggingface.co/spaces/genai-impact/ecologits-calculator)**: EcoLogits is a python library that tracks the energy consumption and environmental footprint of using generative AI models through APIs.
- **[Perplexity Kernels](https://github.com/ppl-ai/pplx-kernels)**: GPU kernels by Perplexity.
- **[Fast Tokenizer](https://github.com/NLPOptimize/flash-tokenizer)**: Fast tokenizer is an efficient and optimized tokenizer engine for llm inference serving.
- **[WeightWatcher](https://github.com/CalculatedContent/WeightWatcher)**: WeightWatcher (WW) is an open-source, diagnostic tool for analyzing Deep Neural Networks (DNN), without needing access to training or even test data..
- **[Cockpit](https://github.com/f-dangel/cockpit)**: A Practical Debugging Tool for Training Deep Neural Networks.
- **[Electrictiy Map](https://app.electricitymaps.com/map/72h/hourly)**: A live map showing the origin of the electricity in world regions and their CO2 intensity.
- **[MLCA](https://github.com/blubrom/MLCA)**: A tool for machine learning life cycle assessment.
- **[TritonParse](https://github.com/pytorch-labs/tritonparse)**: A visualization and analysis tool for Triton IR files, designed to help developers analyze, debug, and understand Triton kernel compilation processes.
- **[Routing on Random Forests](https://github.com/Not-Diamond/RoRF/)**: A framework for training and serving LLM based on random forest-based routers, thus allowing to optimize for costs.
- **[LLMCache](https://github.com/LMCache/LMCache)**: An LLM serving engine extension to reduce time-to-first-token and increase throughput, especially under long-context scenarios. 
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)**: An optimized quantization and inference library for running LLMs locally on modern consumer-class GPUs.
- **[FlashDeBERTa](https://github.com/Knowledgator/FlashDeBERTa)**: Flash implementation of DeBERTa disentangled attention mechanism.
- **[QuACK](https://github.com/Dao-AILab/quack)**: An assortiment of Kernels for GPUs.
- **[Pi-Quant](https://github.com/PrimeIntellect-ai/pi-quant)**: An assortiment of Kernels for CPUs.
- **[pplx-kernels](https://github.com/ppl-ai/pplx-kernels)**: An assortiment of Kernels for GPUs.
- **[LMCache](https://github.com/LMCache/LMCache)**: an LLM serving engine extension to reduce TTFT and increase throughput, especially under long-context scenarios, by optimizing the KV caches.
- **[FastWan](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/)**: a family of video generation models trained via “sparse distillation”.
- **[GEAK Agent](https://github.com/AMD-AIG-AIMA/GEAK-agent)**: This is an LLM-based multi-agent framework, which can generate functional and efficient gpu kernels automatically.
- **[Fused Kernel Library](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary)**: Implementation of a package that allows user to define GPU kernel fusion, for non CUDA programmers.
- **[FlashPack](https://github.com/fal-ai/flashpack)**: a library with efficient Disk-to-GPU Tensor loading.

---

## News Articles 📰
- *"[Responding to the climate impact of generative AI](https://news.mit.edu/2025/responding-to-generative-ai-climate-impact-0930)"* (2025) - MIT News
- *"[Energy and AI Observatory](https://www.iea.org/data-and-statistics/data-tools/energy-and-ai-observatory?tab=Energy+for+AI)"* (2025) - International Energy Agency
- *"[AI’s Impacts, how to limit them, and why](https://bettertech.blog/2025/04/19/ais-impacts-how-to-limit-them-and-why/)"* (2025) - Better Tech
- *"[How much energy does ChatGPT use?](https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use)"* (2025) - Epoch AI
- *"[Data centers et intelligence artificielle : la course au gigantisme](https://www.lemonde.fr/economie/article/2025/06/01/data-centers-et-intelligence-artificielle-la-course-au-gigantisme_6609751_3234.html)"* (2025) - Le Monde
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
- *"[AI in 2030](https://epoch.ai/files/AI_2030.pdf)"* (2025) - Epoch AI
- *"[Intelligence artificielle, données, calculs : quelles infrastructures dans un monde décarboné ?](https://theshiftproject.org/publications/intelligence-artificielle-centres-de-donnees-rapport-final/)"* (2025) - Shift Project
- *"[The environmental impact of local text AI](https://greenspector.com/en/the-environmental-impact-of-local-text-ai/)"* (2025) - Green Spector
- *"[Misinformation by Omission: The Need for More Environmental Transparency in AI](https://arxiv.org/pdf/2506.15572)"* (2025) - None
- *"[A General Framework for Frugal AI](https://www.afnor.org/en/news/referential-for-measuring-and-reducing-environmental-impact-of-ia/)"* (2025) - AFNOR
- *"[The 2025 AI Index Report](https://hai.stanford.edu/ai-index/2025-ai-index-report)"* (2025) - Stanford Human-centered Artificial Intelligence
- *"[Energy and AI](https://www.iea.org/reports/energy-and-ai)"* (2025) - International Energy Agency
- *"[Key challenges for the environmental performance of AI](https://www.sustainableaicoalition.org/key-challenges/)"* (2025) - French Ministry
- *"[Artificial Intelligence and electricity: A system dynamics approach](https://www.se.com/ww/en/insights/sustainability/sustainability-research-institute/artificial-intelligence-electricity-system-dynamics-approach/)"* (2024) - Schneider
- *"[Notable AI Models](https://epoch.ai/data/notable-ai-models)"* (2025) - Epoch AI
- *"[Powering Artificial Intelligence](Deloitte)"* (2024) - Deloitte
- *"[Google Sustainability Reports](https://sustainability.google/reports/)"* (2024) - Google
- *"[How much water does AI consume? The public deserves to know](https://oecd.ai/en/wonk/how-much-water-does-ai-consume)"* (2023) - OECD
- *"[Measuring the environmental impacts of artificial intelligence compute and applications](https://www.oecd.org/en/publications/measuring-the-environmental-impacts-of-artificial-intelligence-compute-and-applications_7babf571-en.html)"* (2022) - OECD

---

## Research Articles 📄
| Paper | Year | Venue | Tags |
|-------|------|-------|------|
| <sub><b>[TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs](https://arxiv.org/pdf/2511.13223)</b></sub> | 2025 | NeurIPS | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[Frequency-Aware Token Reduction for Efficient Vision Transformer](https://arxiv.org/pdf/2511.21477)</b></sub> | 2025 | NeurIPS | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Language Models (Mostly) Know When to Stop Reading](https://arxiv.org/pdf/2502.01025)</b></sub> | 2025 | NeurIPS | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[Why 1 + 1 &lt; 1 in Visual Token Pruning](https://arxiv.org/pdf/2505.10118)</b></sub> | 2025 | NeurIPS | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Don’t Think Longer, Think Wisely: Optimizing Thinking Dynamics for Large Reasoning Models](https://arxiv.org/pdf/2505.21765)</b></sub> | 2025 | NeurIPS | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[The Overthinker’s DIET: Cutting Token Calories with Difficulty-Aware Training](https://arxiv.org/pdf/2505.19217)</b></sub> | 2025 | NeurIPS | <sub><b>![Training](https://img.shields.io/badge/Training-purple)</b></sub> |
| <sub><b>[R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing](https://arxiv.org/pdf/2505.21600)</b></sub> | 2025 | NeurIPS | <sub><b>![SpecDec](https://img.shields.io/badge/SpecDec-purple)</b></sub> |
| <sub><b>[Training Language Models to Reason Efficiently](https://arxiv.org/pdf/2502.04463)</b></sub> | 2025 | NeurIPS | <sub><b>![Training](https://img.shields.io/badge/Training-purple)</b></sub> |
| <sub><b>[AccuQuant: Simulating Multiple Denoising Steps for Quantizing Diffusion Models](https://arxiv.org/pdf/2510.20348)</b></sub> | 2025 | NeurIPS | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[FP4 All the Way: Fully Quantized Training of LLMs](https://arxiv.org/pdf/2505.19115)</b></sub> | 2025 | NeurIPS | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Quartet: Native FP4 Training Can Be Optimal for Large Language Models](https://arxiv.org/html/2505.14669v3)</b></sub> | 2025 | NeurIPS | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization](https://arxiv.org/pdf/2511.04063)</b></sub> | 2025 | NeurIPS | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[DFloat11: Lossless Compression of LLMs and Diffusion Models for Efficient GPU Inference](https://github.com/LeanModels/DFloat11)</b></sub> | 2024 | NeurIPS | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[A Token is Worth over 1,000 Tokens](https://arxiv.org/pdf/2505.12781)</b></sub> | 2025 | NeurIPS | <sub><b>![Distillation](https://img.shields.io/badge/Distillation-purple)</b></sub> |
| <sub><b>[Why Knowledge Distillation Works in Generative Models](https://arxiv.org/pdf/2505.13111)</b></sub> | 2025 | NeurIPS | <sub><b>![Distillation](https://img.shields.io/badge/Distillation-purple)</b></sub> |
| <sub><b>[Knowledge Distillation Detection for Open-Weights Models](https://arxiv.org/pdf/2510.02302)</b></sub> | 2025 | NeurIPS | <sub><b>![Distillation](https://img.shields.io/badge/Distillation-purple)</b></sub> |
| <sub><b>[FFN Fusion: Rethinking Sequential Computation in Large Language Models](https://openreview.net/attachment?id=XUmGMBRv4M&name=pdf)</b></sub> | 2024 | NeurIPS | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Sign-In to the Lottery: Reparameterized Sparse Training](https://arxiv.org/pdf/2504.12801)</b></sub> | 2025 | NeurIPS | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Differentiable Sparsity via D-Gating](https://arxiv.org/pdf/2509.23898)</b></sub> | 2025 | NeurIPS | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[The Graphon Limit Hypothesis](https://arxiv.org/pdf/2510.17515)</b></sub> | 2025 | NeurIPS | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Týr-the-Pruner: Structural Pruning LLMs via Global Sparsity Distribution Optimization](https://arxiv.org/pdf/2503.09657)</b></sub> | 2025 | NeurIPS | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](https://arxiv.org/pdf/2506.20194)</b></sub> | 2025 | NeurIPS | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Spark Transformer: Reactivating Sparsity in FFN and Attention](https://arxiv.org/pdf/2506.06644)</b></sub> | 2025 | NeurIPS | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Multi-Token Prediction Needs Registers](https://arxiv.org/pdf/2505.10518)</b></sub> | 2025 | NeurIPS | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[Q3R: Quadratic Reweighted Rank Regularizer for Effective Low-Rank Training](https://arxiv.org/pdf/2511.04485)</b></sub> | 2025 | NeurIPS | <sub><b>![Factorization](https://img.shields.io/badge/Factorization-purple)</b></sub> |
| <sub><b>[Accurate and Efficient Low-Rank Model Merging in Core Space](https://arxiv.org/pdf/2509.17786)</b></sub> | 2025 | NeurIPS | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)</b></sub> |
| <sub><b>[FALQON: Accelerating LoRA Fine-tuning with Low-Bit Floating-Point Arithmetic](https://arxiv.org/pdf/2510.24061)</b></sub> | 2025 | NeurIPS | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)</b></sub> |
| <sub><b>[LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/pdf/2410.21228)</b></sub> | 2024 | NeurIPS | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)</b></sub> |
| <sub><b>[DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](https://arxiv.org/abs/2509.22793)</b></sub> | 2025 | NeurIPS | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)</b></sub> |
| <sub><b>[KLASS: KL-Guided Fast Inference in Masked Diffusion Models](https://arxiv.org/pdf/2511.05664)</b></sub> | 2025 | NeurIPS | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion](https://github.com/KSH00906/ScaleDiff)</b></sub> | 2025 | NeurIPS | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[AI Should Sense Better, Not Just Scale Bigger: Adaptive Sensing as a Paradigm Shift](https://arxiv.org/pdf/2507.07820)</b></sub> | 2025 | NeurIPS | <sub><b>![Scalability](https://img.shields.io/badge/Scalability-purple)</b></sub> |
| <sub><b>[A Sustainable AI Economy Needs Data Deals That Work for Generators](https://openreview.net/pdf?id=mdKzkjY1dM)</b></sub> | 2024 | NeurIPS | <sub><b>![Sustainability](https://img.shields.io/badge/Sustainability-purple)</b></sub> |
| <sub><b>[We Should Chart an Atlas of All the World's Models](https://horwitz.ai/model-atlas)</b></sub> | 2024 | NeurIPS |  |
| <sub><b>[Noise Hypernetworks: Amortizing Test-Time Compute in Diffusion Models](https://arxiv.org/pdf/2508.09968)</b></sub> | 2025 | NeurIPS | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[Position: Require Frontier AI Labs To Release Small "Analog" Models](https://arxiv.org/pdf/2510.14053)</b></sub> | 2025 | NeurIPS | <sub><b>![Sustainability](https://img.shields.io/badge/Sustainability-purple)</b></sub> |
| <sub><b>[Accelerating Diffusion LLMs via Adaptive Parallel Decoding](https://arxiv.org/pdf/2506.00413)</b></sub> | 2025 | NeurIPS | <sub><b>![SpecDec](https://img.shields.io/badge/SpecDec-purple)</b></sub> |
| <sub><b>[LongLive: Real-time Interactive Long Video Generation](https://arxiv.org/pdf/2509.22622)</b></sub> | 2025 | None | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[MagCache: Fast Video Generation with Magnitude-Aware Cache](https://arxiv.org/pdf/2506.09045)</b></sub> | 2025 | NeurIPS | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[Toward Efficient Inference for Mixture of Experts](https://proceedings.neurips.cc/paper_files/paper/2024/file/98bf3b8505c611ac21055dd9d355c66e-Paper-Conference.pdf)</b></sub> | 2025 | NeurIPS | <sub><b></b></sub> |
| <sub><b>[Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](https://arxiv.org/pdf/2504.09629)</b></sub> | 2025 | NeurIPS | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[DC-VideoGen: Efficient Video Generation with Deep Compression Video Autoencoder](https://arxiv.org/pdf/2509.25182)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation](https://arxiv.org/pdf/2505.18875)</b></sub> | 2025 | NeurIPS | <sub><b></b></sub> |
| <sub><b>[More than Carbon: Cradle-to-Grave environmental impacts of GenAI training on the Nvidia A100 GPU](https://arxiv.org/pdf/2509.00093)</b></sub> | 2025 | None | <sub><b>![Sustainability](https://img.shields.io/badge/Sustainability-purple)</b></sub> |
| <sub><b>[Does Efficiency Lead to Green Machine Learning Model Training? Analyzing Historical Trends in Impacts from Hardware, Algorithmic and Carbon Optimizations](https://hal.science/hal-04839926v4/document)</b></sub> | 2025 | None | <sub><b>![Sustainability](https://img.shields.io/badge/Sustainability-purple) ![Training](https://img.shields.io/badge/Training-purple)</b></sub> |
| <sub><b>[Measuring the environmental impact of delivering AI at Google Scale](https://services.google.com/fh/files/misc/measuring_the_environmental_impact_of_delivering_ai_at_google_scale.pdf)</b></sub> | 2025 | None | <sub><b>![Sustainability](https://img.shields.io/badge/Sustainability-purple)</b></sub> |
| <sub><b>[GANQ: GPU-Adaptive Non-Uniform Quantization for Large Language Models](https://arxiv.org/abs/2501.12956)</b></sub> | 2025 | ICML | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation](https://arxiv.org/pdf/2506.00329)</b></sub> | 2025 | None | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[HiCache: Training-free Acceleration of Diffusion Models via Hermite Polynomial-based Feature Caching](https://arxiv.org/pdf/2508.16984)</b></sub> | 2025 | ICML | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[ERTACache: Error Rectification and Timesteps Adjustment for Efficient Diffusion](https://arxiv.org/pdf/2508.21091)</b></sub> | 2025 | None | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[DiCache: Let Diffusion Model Determine its Own Cache]()</b></sub> | 2025 | None | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[Set Block Decoding is a Language Model Inference Accelerator](https://arxiv.org/pdf/2509.04185)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Position: Small Language Models are the Future of Agentic AI](https://arxiv.org/pdf/2506.02153)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Fast-dLLM v2: Efficient Block-Diffusion Large Language Model](https://nvlabs.github.io/Fast-dLLM/v2/)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](https://arxiv.org/pdf/2503.09641)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models](https://arxiv.org/pdf/2505.19223)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Let LLM Tell What to Prune and How Much to Prune](https://openreview.net/pdf?id=zFR5aWGaUs)</b></sub> | 2025 | ICML | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[SlimLLM: Accurate Structured Pruning for Large Language Models](https://arxiv.org/pdf/2505.22689)</b></sub> | 2025 | ICML | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation](https://openreview.net/pdf?id=bzbuZ0ItBq)</b></sub> | 2025 | ICML | <sub><b>![SpecDec](https://img.shields.io/badge/SpecDec-purple)</b></sub> |
| <sub><b>[OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM inference](https://arxiv.org/pdf/2507.03865)</b></sub> | 2025 | ICML | <sub><b></b></sub> |
| <sub><b>[SkipGPT: Each Token is One of a Kind](https://openreview.net/pdf?id=d7v2iUSa9s)</b></sub> | 2025 | ICML | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[AdaSplash: Adaptive Sparse Flash Attention](https://openreview.net/pdf?id=OWIPDWhUcO)</b></sub> | 2025 | ICML | <sub><b></b></sub> |
| <sub><b>[Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression](https://openreview.net/pdf?id=rkwXYSDKso)</b></sub> | 2025 | ICML | <sub><b></b></sub> |
| <sub><b>[An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks](https://openreview.net/pdf?id=Nvf4jFsbv9)</b></sub> | 2025 | ICML | <sub><b></b></sub> |
| <sub><b>[Accelerating Large Language Model Reasoning via Speculative Search](https://openreview.net/pdf?id=oq0t5BXilT)</b></sub> | 2025 | ICML | <sub><b>![SpecDec](https://img.shields.io/badge/SpecDec-purple)</b></sub> |
| <sub><b>[ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering](https://openreview.net/pdf?id=HZCx5EToh9)</b></sub> | 2025 | ICML | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[HarmoniCa: Harmonizing Training and Inference for Better Feature Caching in Diffusion Transformer Acceleration](https://openreview.net/pdf?id=NQyqpK6d72)</b></sub> | 2025 | ICML | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[Attention-Level Speculation](https://openreview.net/pdf?id=4OszSYdsgO)</b></sub> | 2025 | ICML | <sub><b>![SpecDec](https://img.shields.io/badge/SpecDec-purple)</b></sub> |
| <sub><b>[EvoPress: Accurate Dynamic Model Compression via Evolutionary Search](https://openreview.net/pdf?id=l7QzcZpjc5)</b></sub> | 2025 | ICML | <sub><b></b></sub> |
| <sub><b>[Hardware and Software Platform Inference](https://openreview.net/pdf?id=kdmjVF1iDO)</b></sub> | 2025 | ICML | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[MoH: Multi-Head Attention as Mixture-of-Head Attention](https://openreview.net/pdf?id=eYtgs9k75o)</b></sub> | 2025 | ICML | <sub><b></b></sub> |
| <sub><b>[QoS-Efficient Serving of Multiple Mixture-of-Expert LLMs Using Partial Runtime Reconfiguration](https://openreview.net/pdf?id=g45SHBmZLz)</b></sub> | 2025 | ICML | <sub><b></b></sub> |
| <sub><b>[KV Cache Compression via Sparse Coding over Universal Dictionaries](https://openreview.net/pdf?id=Yh9vxlxnjA)</b></sub> | 2025 | ICML | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference](https://openreview.net/pdf?id=zDwipF6h06)</b></sub> | 2025 | ICML | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference](https://openreview.net/pdf?id=Y0yXuQtPn8)</b></sub> | 2025 | ICML | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[any4: Learned 4-bit Numeric Representation for LLMs](https://openreview.net/pdf?id=tJmhOPkWCj)</b></sub> | 2025 | ICML | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[AdaDecode: Accelerating LLM Decoding with Adaptive Layer Parallelism](https://openreview.net/pdf?id=VnO2GEpmlb)</b></sub> | 2025 | ICML | <sub><b>![SpecDec](https://img.shields.io/badge/SpecDec-purple)</b></sub> |
| <sub><b>[BOA: Attention-aware Post-training Quantization without Backpropagation](https://openreview.net/pdf?id=Uvj6XcSJ5d)</b></sub> | 2025 | ICML | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Radio: Rate–Distortion Optimization for Large Language Model Compression](https://openreview.net/pdf?id=ifnxXCCEiM)</b></sub> | 2025 | ICML | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Auditing Prompt Caching in Language Model APIs](https://arxiv.org/pdf/2502.07776)</b></sub> | 2025 | ICML | <sub><b></b></sub> |
| <sub><b>[Mind the Gap: A Practical Attack on GGUF Quantization](https://www.arxiv.org/pdf/2505.23786)</b></sub> | 2025 | ICML | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Olica: Efficient Structured Pruning of Large Language Models without Retraining](https://openreview.net/pdf?id=hhhcwCgyM1)</b></sub> | 2025 | ICML | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[TESS 2: A Large-Scale Generalist Diffusion Language Model](https://arxiv.org/abs/2502.13917)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference](https://arxiv.org/abs/2508.02193)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential](https://arxiv.org/html/2507.11851v1)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/pdf/2404.19737)</b></sub> | 2024 | None | <sub><b></b></sub> |
| <sub><b>[The Fused Kernel Library: A C++ API to Develop Highly-Efficient GPU Libraries](https://arxiv.org/pdf/2508.07071)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[QuarterMap: Efficient Post-Training Token Pruning for Visual State Space Models](https://openreview.net/attachment?id=pGJB9zGGwC&name=pdf)</b></sub> | 2025 | None | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Fast Video Generation with Sliding Tile Attention](https://arxiv.org/abs/2502.04507)</b></sub> | 2025 | ICML | <sub><b></b></sub> |
| <sub><b>[Quartet: Native FP4 Training Can Be Optimal for Large Language Models](https://arxiv.org/abs/2505.14669)</b></sub> | 2025 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference](https://arxiv.org/abs/2505.09598)</b></sub> | 2025 | None | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[MagCache: Fast Video Generation with Magnitude-Aware Cache](https://arxiv.org/pdf/2506.09045)</b></sub> | 2025 | None | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[Compressing Language Models for Specialized Domains](https://arxiv.org/abs/2502.18424)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training](https://arxiv.org/abs/2505.11594)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[XAttention: Block Sparse Attention with Antidiagonal Scoring](https://hanlab.mit.edu/projects/xattention)</b></sub> | 2025 | ICML | <sub><b></b></sub> |
| <sub><b>[Jenga: Effective Memory Management for Serving LLM with Heterogeneity](https://arxiv.org/abs/2503.18292)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Learning Few-Step Diffusion Models by Trajectory Distribution Matching](https://arxiv.org/abs/2503.06674)</b></sub> | 2025 | ICCV | <sub><b></b></sub> |
| <sub><b>[Radial Attention: O(nlogn) Sparse Attention with Energy Decay for Long Video Generation](https://arxiv.org/abs/2506.19852)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://nvlabs.github.io/Fast-dLLM/paper/fast_dllm.pdf)</b></sub> | 2025 | None | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[Chipmunk: Training-Free Acceleration of Diffusion Transformers with Dynamic Column-Sparse Deltas](https://arxiv.org/abs/2506.03275)</b></sub> | 2025 | None | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[Mirage: A Multi-Level Superoptimizer for Tensor Programs](https://arxiv.org/pdf/2405.05751)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization](https://arxiv.org/abs/2505.06371)</b></sub> | 2025 | None | <sub><b>![Sustainability](https://img.shields.io/badge/Sustainability-purple)</b></sub> |
| <sub><b>[AB-Cache: Training-Free Acceleration of Diffusion Models via Adams-Bashforth Cached Feature Reuse](https://arxiv.org/pdf/2504.10540)</b></sub> | 2025 | None | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[Hardware-Efficient Attention for Fast Decoding](https://arxiv.org/abs/2505.21487)</b></sub> | 2025 | None | <sub><b>![Hardware](https://img.shields.io/badge/Hardware-purple)</b></sub> |
| <sub><b>[Model-Preserving Adaptive Rounding](https://arxiv.org/pdf/2505.22988v1)</b></sub> | 2025 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Frugal AI: Introduction, Concepts, Development and Open Questions](https://www.researchgate.net/publication/390920260_Frugal_AI_Introduction_Concepts_Development_and_Open_Questions)</b></sub> | 2025 | None | <sub><b>![Sustainability](https://img.shields.io/badge/Sustainability-purple)</b></sub> |
| <sub><b>[Making AI Less “Thirsty”: Uncovering and Addressing the Secret Water Footprint of AI Models](https://arxiv.org/pdf/2304.03271)</b></sub> | 2025 | None | <sub><b>![Sustainability](https://img.shields.io/badge/Sustainability-purple)</b></sub> |
| <sub><b>[Efficient Time Series Processing for Transformers and State-Space Models through Token Merging](https://arxiv.org/pdf/2405.17951)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[A Survey on Inference Engines for Large Language Models: Perspectives on Optimization and Efficiency](https://arxiv.org/abs/2505.01658)</b></sub> | 2025 | None | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference](https://arxiv.org/abs/2502.18137)</b></sub> | 2025 | None | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)</b></sub> | 2025 | None | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
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
| <sub><b>[FlexiDiT: Your Diffusion Transformer Can Easily Generate High-Quality Samples with Less Compute](https://arxiv.org/abs/2502.20126)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling](https://arxiv.org/pdf/2502.06703)</b></sub> | 2025 | None | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/pdf/2405.16406)</b></sub> | 2025 | ICLR | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Making AI Less “Thirsty”: Uncovering and Addressing the Secret Water Footprint of AI Models](https://arxiv.org/pdf/2304.03271)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps](https://arxiv.org/pdf/2501.09732)</b></sub> | 2025 | None | <sub><b>![Inference](https://img.shields.io/badge/Inference-purple)</b></sub> |
| <sub><b>[QuEST: Stable Training of LLMs with 1-Bit Weights and Activations](https://arxiv.org/pdf/2502.05003)</b></sub> | 2025 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Distillation Scaling Laws](https://arxiv.org/pdf/2502.08606)</b></sub> | 2025 | None | <sub><b>![Distillation](https://img.shields.io/badge/Distillation-purple)</b></sub> |
| <sub><b>[From Efficiency Gains to Rebound Effects: The Problem of Jevons' Paradox in AI's Polarized Environmental Debate](https://arxiv.org/abs/2501.16548v1)</b></sub> | 2025 | None | <sub><b></b></sub> |
| <sub><b>[Coca4ai: checking energy behaviors on AI data centers](https://arxiv.org/abs/2407.15670)</b></sub> | 2024 | None | <sub><b>![Sustainability](https://img.shields.io/badge/Sustainability-purple) ![Scalability](https://img.shields.io/badge/Scalability-purple)</b></sub> |
| <sub><b>[Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514)</b></sub> | 2024 | None | <sub><b></b></sub> |
| <sub><b>[LTX-Video: Realtime Video Latent Diffusion](https://arxiv.org/abs/2501.00103)</b></sub> | 2024 | CVPR | <sub><b></b></sub> |
| <sub><b>[How Green Can AI Be? A Study of Trends in Machine Learning Environmental Impacts](https://arxiv.org/abs/2412.17376)</b></sub> | 2024 | None | <sub><b>![Sustainability](https://img.shields.io/badge/Sustainability-purple)</b></sub> |
| <sub><b>[QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/pdf/2404.00456)</b></sub> | 2024 | NeurIPS | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[The Iterative Optimal Brain Surgeon: Faster Sparse Recovery by Leveraging Second-Order Information](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fc1dce23e0ba01d3b691789cfd4f65c3-Abstract-Conference.html)</b></sub> | 2024 | NeurIPS | <sub><b></b></sub> |
| <sub><b>[Palu: Compressing KV-Cache with Low-Rank Projection](https://arxiv.org/abs/2407.21118)</b></sub> | 2024 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
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
| <sub><b>[Hype, Sustainability, and the Price of the Bigger-is-Better Paradigm in AI](https://arxiv.org/pdf/2409.14160)</b></sub> | 2024 | None | <sub><b>![Sustainability](https://img.shields.io/badge/Sustainability-purple)</b></sub> |
| <sub><b>[Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes](https://arxiv.org/abs/2402.05406)</b></sub> | 2024 | ICLR | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Exploring the Carbon Footprint of Hugging Face's ML Models: A Repository Mining Study](https://www.researchgate.net/publication/375508988_Exploring_the_Carbon_Footprint_of_Hugging_Face's_ML_Models_A_Repository_Mining_Study)</b></sub> | 2023 | ESEM | <sub><b>![Sustainability](https://img.shields.io/badge/Sustainability-purple)</b></sub> |
| <sub><b>[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)</b></sub> | 2023 | SOSP | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[Broken Neural Scaling Laws](https://arxiv.org/abs/2210.14891)</b></sub> | 2023 | ICLR | <sub><b></b></sub> |
| <sub><b>[Model Compression in Practice: Lessons Learned from Practitioners Creating On-device Machine Learning Experiences](https://arxiv.org/html/2310.04621v2)</b></sub> | 2023 | None |  |
| <sub><b>[Post Training Mixed Precision Quantization of Neural Networks using First-Order Information](https://openaccess.thecvf.com/content/ICCV2023W/RCV/papers/Chauhan_Post_Training_Mixed_Precision_Quantization_of_Neural_Networks_Using_First-Order_ICCVW_2023_paper.pdf)</b></sub> | 2023 | ICCV | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/pdf/2310.01889)</b></sub> | 2023 | None | <sub><b></b></sub> |
| <sub><b>[A Practical Mixed Precision Algorithm for Post-Training Quantization](https://arxiv.org/abs/2302.05397)</b></sub> | 2023 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)</b></sub> | 2023 | ICML | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs](https://arxiv.org/pdf/2312.15230)</b></sub> | 2023 | None | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Trends in AI inference energy consumption: Beyond the performance-vs-parameter laws of deep learning](https://www.sciencedirect.com/science/article/pii/S2210537923000124#:~:text=However%2C%20for%20deployed%20systems%2C%20inference,but%20inference%20is%20done%20repeatedly.)</b></sub> | 2023 | Sustainable Computing: Informatics and Systems | <sub><b>![Sustainability](https://img.shields.io/badge/Sustainability-purple)</b></sub> |
| <sub><b>[An experimental comparison of software-based power meters: focus on CPU and GPU](https://inria.hal.science/hal-04030223v2/file/_CCGrid23__An_experimental_comparison_of_software_based_power_meters__from_CPU_to_GPU.pdf)</b></sub> | 2023 | CCGrid | <sub><b>![Hardware](https://img.shields.io/badge/Hardware-purple)</b></sub> |
| <sub><b>[Fast Inference from Transformers via Speculative Decoding](https://openreview.net/pdf?id=C9NEblP8vS)</b></sub> | 2023 | ICML | <sub><b>![Caching](https://img.shields.io/badge/Caching-purple)</b></sub> |
| <sub><b>[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)</b></sub> | 2023 | ICLR | <sub><b></b></sub> |
| <sub><b>[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/pdf/2210.17323)</b></sub> | 2023 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Mixed-Precision Neural Network Quantization via Learned Layer-wise Importance](https://arxiv.org/pdf/2203.08368)</b></sub> | 2022 | ECCV | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Knowledge Distillation: A Good Teacher is Patient and Consistent](https://arxiv.org/pdf/2106.05237)</b></sub> | 2022 | CVPR | <sub><b>![Distillation](https://img.shields.io/badge/Distillation-purple)</b></sub> |
| <sub><b>[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)</b></sub> | 2022 | ICLR | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)</b></sub> |
| <sub><b>[LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)</b></sub> | 2022 | NeurIPS | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Optimal Clipping and Magnitude-aware Differentiation for Improved Quantization-aware Training](https://arxiv.org/abs/2206.06501)</b></sub> | 2022 | ICML | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Sustainable AI: Environmental Implications, Challenges and Opportunities](https://arxiv.org/abs/2111.00364)</b></sub> | 2022 | None | <sub><b>![Sustainability](https://img.shields.io/badge/Sustainability-purple)</b></sub> |
| <sub><b>[Learnable Lookup Table for Neural Network Quantization](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learnable_Lookup_Table_for_Neural_Network_Quantization_CVPR_2022_paper.pdf)</b></sub> | 2022 | CVPR | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)</b></sub> | 2022 | None | <sub><b></b></sub> |
| <sub><b>[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)</b></sub> | 2022 | None | <sub><b></b></sub> |
| <sub><b>[Towards a Unified View of Parameter-Efficient Transfer Learning](https://arxiv.org/abs/2110.04366)</b></sub> | 2022 | ICLR | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)</b></sub> |
| <sub><b>[Parameter-Efficient Transfer Learning with Diff Pruning](https://arxiv.org/abs/2012.07463)</b></sub> | 2021 | ACL | <sub><b>![PEFT](https://img.shields.io/badge/Peft-purple)![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033)</b></sub> | 2020 | MLSys | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |
| <sub><b>[Scaling Laws for Autoregressive Generative Modeling](https://arxiv.org/abs/2010.14701)</b></sub> | 2020 | None | <sub><b></b></sub> |
| <sub><b>[Model Compression via Distillation and Quantization](https://arxiv.org/abs/1802.05668)</b></sub> | 2018 | ICLR | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-purple)</b></sub> |
| <sub><b>[Optimal Brain Damage](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)</b></sub> | 1989 | NeurIPs | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-purple)</b></sub> |

---

## Blogs 📰
- *"[Vocabulary Trimming: An Easy and Effective Method for SLM Acceleration](https://blog.squeezebits.com/vocabulary-trimming-methods) (2025)"* - Squeeze Bits
- *"[Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm) (2025)"* - Aleksa Gordić
- *"[Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) (2025)"* - Hazy Research
- *"[Our contribution to a global environmental standard for AI](https://mistral.ai/news/our-contribution-to-a-global-environmental-standard-for-ai) (2025)"* - Mistral AI
- *"[AI: It's All About Inference Now](https://queue.acm.org/detail.cfm?id=3733701) (2025)"* - ACM Queue
- *"[ScalarLM vLLM Optimization with Virtual Channels](https://www.scalarlm.com/blog/scalarlm-vllm-optimization-with-virtual-channels/)" (2025)* - ScalarLM
- *"[Review of Inference Optimization](https://www.aussieai.com/research/overview)" (2025)* - Aussie AI
- *"[The Limits of Large Fused Kernels on Nvidia GPUs: Why Real-Time AI Inference Needs More](https://smallest.ai/blog/the-limits-of-large-fused-kernels-on-nvidia-gpus-why-real-time-ai-inference-needs-more)" (2025)* - Smallest AI
- *"[How Much Power does a SOTA Open Video Model Use?](https://huggingface.co/blog/jdelavande/text-to-video-energy-cost)" (2025)* - Hugging Face  
- *"[Improving Quantized FP4 Weight Quality via Logit Distillation](https://mobiusml.github.io/fp4_blogpost/)" (2025)* - Mobius Labs
- *"[Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/?ncid=so-link-105283&linkId=100000370829029)" (2025)* - Nvidia
- *"[The LLM Engineer Almanac](https://modal.com/llm-almanac/advisor?filters=model%3DLlama+3.1+8B%2Ctokens%3D128%3B1024%2Cttft_p95%3C1)" (2025)* - Modal
- *"[Enhance Your Models in 5 Minutes with the Hugging Face Kernel Hub](https://huggingface.co/blog/hello-hf-kernels)" (2025)* - Hugging Face
- *"[Reduce, Reuse, Recycle: Why Open Source is a Win for Sustainability](https://huggingface.co/blog/sasha/reduce-reuse-recycle)" (2025)* - Hugging Face
- *"[Mixture of Experts: When Does It Really Deliver Energy Efficiency?](https://www.neuralwatt.com/blog/mixture-of-experts-when-does-it-really-deliver-energy-efficiency)" (2025)* - Neural Watt
- *"[Efficient and Portable Mixture-of-Experts Communication](https://www.perplexity.ai/fr/hub/blog/efficient-and-portable-mixture-of-experts-communication)" (2025)* - Perplexity
- *"[Optimizing Tokenization for Faster and Efficient LLM Processing](https://medium.com/@harishpillai1994/optimizing-tokenization-for-faster-and-efficient-llm-processing-bdc87b8f9fe3)" (2025)* - Medium
- *"[Tensor Parallelism with CUDA - Multi-GPU Matrix Multiplication](https://substack.com/home/post/p-158663472)" (2025)* - Substack
- *"[Automating GPU Kernel Generation with DeepSeek-R1 and Inference Time Scaling](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/)" (2025)* - Nvidia Developer
- *"[AI CUDA Engineer](https://sakana.ai/ai-cuda-engineer/)" (2025)* - Sakana AI
- *"[The ML/AI Engineer's starter guide to GPU Programming](https://neuralbits.substack.com/p/the-mlai-engineers-starter-guide)" (2025)* - Neural Bits
- *"[A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) (2024)"* - Substack
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
- **AI Efficiency Courses: [Slides](https://ln5.sync.com/dl/7d21bc370/gxpiqj2b-4k22jgex-x8i7zgxr-9pkajy52), [Exercises](https://github.com/PrunaAI/courses)** (2025) - Lecture by Bertrand Charpentier
- **Data Compression, Theory and Applications: [YouTube](https://www.youtube.com/c/MITHANLab), [Slides](https://stanforddatacompressionclass.github.io/notes/contents.html#ee274-data-compression-course-notes)** (2024) - Stanford
- **[MIT Han's Lab](https://www.youtube.com/c/MITHANLab)** (2024) - Lecture by Song Han
- **[GPU Mode](https://www.youtube.com/@GPUMODE)** (2020) - Tutorials by GPU mode community

---

## People 🧑‍💻

| Name                  | Affiliation                       | Research Interests | Social Media |
|-----------------------|-----------------------------------|---------------------|--------------|
| **James Martin**        | Better Tech                        | <sub>AI Sustainability</sub> | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jamesmartin75/) |
| **Saleh Ashkboos**        | ETH Zurich                        | <sub>Quantization</sub> | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/saleh-ashkboos-806628161/) |
| **Dan Alistarh**        | IST Austria                        | <sub>AI Compression</sub> | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dan-alistarh-613ba739/) |
| **Elias Frantar**        | OpenAI                        | <sub>Quantization</sub> | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/elias-frantar-5b43181a4/) |
| **Tim Dettmers**        | CMU                        | <sub>Quantization</sub> | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/timdettmers/) |
| **Song Han**        | MIT                         | <sub>AI Efficiency</sub> | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/songhanmit/) |
| **Scott Chamberlin**        | TBD                         | <sub>AI Efficiency</sub> | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/scott-t-chamberlin/) |
| **Benoit Petit**        | Boavista                         | <sub>Data Center Efficiency</sub> | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bepetit/) |
| **Samuel Rincé**        | Gen AI Impact                                 | <sub>AI Efficiency, Sustainability</sub> | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/samuel-rince/) |
| **Théo Alves Da Costa**           | Ekimetrics                       | <sub>AI Efficiency, Sustainability</sub> | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/th%C3%A9o-alves-da-costa-09397a82/) |
| **Sasha Luccioni**      | Hugging Face                       | <sub>AI Sustainability</sub> | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sashaluccioniphd/) |
| **Anne-Laure Ligozat**      | ENSIEE                       | <sub>AI Sustainability</sub> | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anne-laure-ligozat-009a095/?originalSubdomain=fr) |
| **Boris Gamazaychikov**      | Sales Force | <sub>AI Sustainability</sub>                 | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bgamazay/) |
| **Julie Ravillon**           | Sales Force | <sub>AI Sustainability</sub>           | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/julieravillon/) |
| **Will Alpine**              | Enabled EMissions Camapaigns | <sub>AI Sustainability</sub>       | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/willalpine/) |
| **Holly Alpine** | Enabled EMissions Camapaigns | <sub>AI Sustainability</sub>               | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hollyalpine/) |
| **Drew Wilkinson**           | Climate Leadership Collective | <sub>AI Sustainability</sub>           | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/andrewmartinwilkinson/) |
| **Maren Costa**              | WorkforClimate | <sub>AI Sustainability</sub>             | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/marencosta/) |
| **Lou Welgryn**              | Data4Good | <sub>AI Ethics, Sustainability</sub>             | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lou-welgryn-460434b0/) |
| **Caroline Jean-Pierre**     | Gen AI Impact | <sub>AI Sustainability</sub>         | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/caroline-jean-pierre/) |
| **Claire Saignol**           | Gen AI Impact | <sub>AI Sustianability </sub>              | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/claire-saignol/) |
| **Juliette Fropier**         | French Ministry | <sub>AI Sustainability</sub>             | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/juliette-fropier/) |
| **Helene Costa de Beauregard** | French Ministry | <sub>AI Sustainability</sub>                 | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/helenecostadebeauregard/) |
| **Rémy Marrone**             | Independent | <sub>AI Sustainability</sub>           | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/remymarrone/) |
| **Mark Butcher**             | Positive Cloud | <sub>Cloud sustainability</sub>       | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/markbutcher/) |
| **Robert Keus**              | Green PT | <sub>AI Sustainability</sub>               | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/robertkeus/) |
| **Cas Burggraaf**            | GreenPT | <sub>AI Sustainability</sub>           | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/casburggraaf/) |
| **Wilco Burggraaf**          | GreenPT | <sub>AI Sustainability</sub>             | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/wilco-burggraaf-a6b15517/) |
| **Anna Lerner Nesbitt**      | Climate Collective | <sub>AI Sustainability</sub>              | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anna-lerner-nesbitt/) |
| **Scott Chamberlin**         | Neural Watt | <sub>AI Sustainability</sub>             | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/scott-t-chamberlin/) |
| **Jeremy Tamanini**          | Dual CItizen LLC | <sub>AI Sustainability</sub>                 | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jeremytamanini/) |
| **Emma Strubell**          | CMU | <sub>AI Sustainability</sub>                 | |

## Organizations 🌍

| Organization           | Description                                                              | Website                                              |
|------------------------|--------------------------------------------------------------------------|------------------------------------------------------|
| **Data4Good**           | A platform that connects data scientists with social impact projects to address global challenges using data. | [data4good.org](https://www.data4good.org/)           |
| **Gen AI Impact**           | A platform dedidaceted to understand generative AI environmental footprint. | [genai-impact.org](https://genai-impact.org/)           |
| **Make.org**            | A global platform that empowers citizens to propose and take action on social and environmental issues through collective projects. | [make.org](https://www.make.org/)                     |
| **CodeCarbon**          | A tool that helps track the carbon emissions of machine learning models and optimizes them for sustainability. | [codecarbon.io](https://www.codecarbon.io/)           |
| **Sustainable AI Coalition** | An organization dedicated to advancing sustainability in AI technologies and promoting best practices for green AI. | [sustainableaicoalition.org](https://www.sustainableaicoalition.org/) |
| **FruitPunch AI** | A community that solves AI solutions for impact organizations that contribute to the SDG's. | [fruitpunch.ai](https://www.fruitpunch.ai/) |

---

## Contributing 🤝
Contributions are welcome! Please follow our [contribution guidelines](CONTRIBUTING.md) to add new resources or suggest improvements that promote AI efficiency. Youc can contact @sharpenb if you have any questions.

---

## License 📄
This project is licensed under the [MIT License](LICENSE). Feel free to share and use the resources as needed.
