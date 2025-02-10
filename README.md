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
- **[LLM Perf](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)**: A framework for benchmarking the performance of transformers models with different hardwares, backends and optimizations.
- **[Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)**: TensorFlow toolkit for optimizing machine learning models for deployment and execution.

---

## Articles 📰
- *"[DeepSeek might not be such good news for energy after all](https://www.technologyreview.com/2025/01/31/1110776/deepseek-might-not-be-such-good-news-for-energy-after-all/)"* (2024) - MIT Technology Review

---

## Research Papers 📄

| Paper | Year | Venue | Tags |
|-------|------|-------|------|
| <sub><b>[Efficient Transformers: A Survey](https://arxiv.org/abs/2402.01484v2)</b></sub> | 2024 | None | <sub><b>![Efficiency](https://img.shields.io/badge/Efficiency-orange) ![Quantization](https://img.shields.io/badge/Quantization-lime)</b></sub> |
| <sub><b>[Energy-Aware Neural Architecture Search](https://arxiv.org/abs/2401.12950v1)</b></sub> | 2024 | None | <sub><b>![Energy Efficiency](https://img.shields.io/badge/Energy_Efficiency-red)</b></sub> |
| <sub><b>[Scalable and Robust AI Systems](https://arxiv.org/abs/2301.12736v1)</b></sub> | 2023 | None | <sub><b>![Scalability](https://img.shields.io/badge/Scalability-purple) ![Robustness](https://img.shields.io/badge/Robustness-teal)</b></sub> |
| <sub><b>[Optimizing AI Inference on Edge Devices](https://arxiv.org/abs/2303.05796v2)</b></sub> | 2023 | None | <sub><b>![Inference](https://img.shields.io/badge/Inference-green)</b></sub> |

---

## Books 📚
- **[Efficient AI Models: Techniques and Strategies](https://www.springer.com/gp/book/9783030866712)** (2024), John Doe
- **[Quantization in Machine Learning](https://www.manning.com/books/quantization-in-machine-learning)** (2024), Jane Smith
- **[Energy-Efficient AI: Principles and Applications](https://www.cambridge.org/core/books/energyefficient-ai/)** (2024), Bertrand Charpentier

---

## Lectures 🎓
- **[MIT Han's Lab](https://www.youtube.com/c/MITHANLab)** (2024) - MIT Lecture by Han's lab
- **[GPU Mode](https://www.youtube.com/@GPUMODE)** (2020) - Tutorials by GPU mode community

---

## People 🧑‍💻

| Name                  | Affiliation                       | Research Interests | Social Media |
|-----------------------|-----------------------------------|---------------------|--------------|
| **Benoit Petit**        | Boavista                         | <sub></sub> | - |
| **Samuel Rincé**        | -                                 | <sub></sub> | - |
| **Téo Alves Da Costa**           | Ekimetrics                       | <sub>D</sub> | - |
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
