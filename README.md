# 🌟 Awesome AI Efficiency 🌟

![Awesome](https://awesome.re/badge.svg) ![MIT License](https://img.shields.io/badge/license-MIT-brightgreen)

A curated list of resources dedicated to enhancing efficiency in AI systems. This repository covers a wide range of topics essential for optimizing AI models and processes, aiming to make AI faster, cheaper, smaller, and greener!

### Topics Summary 🎨

| Topic            | Description                                    | Badge Example                                          |
|-------------------|------------------------------------------------|-------------------------------------------------------|
| **Quantization**  | Reducing precision of AI models without loss  | ![Quantization](https://img.shields.io/badge/Quantization-lime) |
| **Pruning**       | Removing unnecessary model parameters for efficiency | ![Pruning](https://img.shields.io/badge/Pruning-orange) |
| **Caching**       | Storing computation results for faster reuse  | ![Caching](https://img.shields.io/badge/Caching-green) |
| **Distillation**  | Transferring knowledge from a large model to a smaller one | ![Distillation](https://img.shields.io/badge/Distillation-blue) |
| **Factorization** | Breaking down complex models into simpler, efficient components | ![Factorization](https://img.shields.io/badge/Factorization-purple) |
| **Compilation**   | Optimizing model code for specific hardware and environments | ![Compilation](https://img.shields.io/badge/Compilation-red) |
| **Hardware**      | Leveraging specialized hardware for faster model execution | ![Hardware](https://img.shields.io/badge/Hardware-teal) |
| **Training**      | Techniques for making model training faster and more efficient | ![Training](https://img.shields.io/badge/Training-orange) |
| **Inference**     | Optimizing the speed and resource usage during model inference | ![Inference](https://img.shields.io/badge/Inference-lime) |
| **Sustainability** | Strategies to reduce the environmental impact of AI systems | ![Sustainability](https://img.shields.io/badge/Sustainability-blue) |
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

---

## Tools 🛠️
- :heart: **[Pruna](https://docs.pruna.ai/en/latest/)** :heart:: A package to make AI models faster, smaller, faster, greener by combining compression methods (incl. quantization, pruning, caching, compilation, distillation...) on various hardware.
- **[TensorRT](https://developer.nvidia.com/tensorrt)**: High-performance deep learning inference library for NVIDIA GPUs.
- **[ONNX](https://onnx.ai/)**: Open Neural Network Exchange format for interoperability among deep learning frameworks.
- **[Code Carbon](https://mlco2.github.io/codecarbon/)**: Library to track energy and carbon efficiency of various hardware.
- **[LLM Perf](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)**: A framework for benchmarking the performance of transformers models with different hardwares, backends and optimizations.
- **[AI Energy Score](https://huggingface.co/spaces/AIEnergyScore/submission_portal)**: An initiative to establish comparable energy efficiency ratings for AI models, helping the industry make informed decisions about sustainability in AI development.
- **[Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)**: TensorFlow toolkit for optimizing machine learning models for deployment and execution.

---

## Articles 📰
- *"[DeepSeek might not be such good news for energy after all](https://www.technologyreview.com/2025/01/31/1110776/deepseek-might-not-be-such-good-news-for-energy-after-all/)"* (2024) - MIT Technology Review

---

## Research Papers 📄

| Paper | Year | Venue | Tags |
|-------|------|-------|------|
| <sub><b>[Distillation Scaling Laws]([https://arxiv.org/pdf/2502.08606))</b></sub> | 2025 | None | <sub><b>![Distillation](https://img.shields.io/badge/Distillation-blue)</b></sub> |
| <sub><b>[Pushing the Limits of Large Language Model Quantization via the Linearity Theorem](https://arxiv.org/abs/2411.17525)</b></sub> | 2024 | None | <sub><b>![Quantization](https://img.shields.io/badge/Quantization-lime)</b></sub> |
| <sub><b>[DeepCache: Accelerating Diffusion Models for Free](https://arxiv.org/pdf/2312.00858)</b></sub> | 2024 | CVPR | <sub><b>![Caching](https://img.shields.io/badge/Caching-green)</b></sub> |
| <sub><b>[Fast Inference from Transformers via Speculative Decoding]([https://arxiv.org/pdf/2312.00858](https://openreview.net/pdf?id=C9NEblP8vS))</b></sub> | 2023 | ICML | <sub><b>![Caching](https://img.shields.io/badge/Caching-green)</b></sub> |
| <sub><b>[Knowledge distillation: A good teacher is patient and consistent]([https://arxiv.org/pdf/2106.05237))</b></sub> | 2022 | CVPR | <sub><b>![Distillation](https://img.shields.io/badge/Distillation-blue)</b></sub> |
| <sub><b>[Optimal Brain Damage](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)</b></sub> | 1989 | NeurIPs | <sub><b>![Pruning](https://img.shields.io/badge/Pruning-orange)</b></sub> |

---

## Books 📚
- **[Programming Massively Parallel Processors: A Hands-on Approach](https://www.amazon.com/dp/0323912311?ref_=cm_sw_r_cp_ud_dp_YVNSMFJMGQ9N457Z8Q6D)** (2022), Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj


---

## Lectures 🎓
- **[MIT Han's Lab](https://www.youtube.com/c/MITHANLab)** (2024) - MIT Lecture by Han's lab
- **[GPU Mode](https://www.youtube.com/@GPUMODE)** (2020) - Tutorials by GPU mode community

---

## People 🧑‍💻

| Name                  | Affiliation                       | Research Interests | Social Media |
|-----------------------|-----------------------------------|---------------------|--------------|
| **Benoit Petit**        | Boavista                         | <sub>Data Center Efficiency</sub> | - |
| **Samuel Rincé**        | Gen AI Impact                                 | <sub>AI Efficiency, Sustainability</sub> | - |
| **Téo Alves Da Costa**           | Ekimetrics                       | <sub>AI Efficiency, Sustainability</sub> | - |
| **Sasha Luccioni**      | Hugging Face                       | <sub>AI Efficiency, Sustainability</sub> | - |

## Organizations 🌍

## Relevant Organizations 🌍

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
