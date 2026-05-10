# Explainable Urban Metro Flow Prediction via Spatio-Temporal Graph Learning and Large Language Models  

This repository contains the official implementation of **MFP-LLM**, a novel framework for **urban metro passenger flow prediction** that integrates **spatio-temporal graph learning** with **instruction-driven large language models (LLMs)**.  

🚇 The goal is to achieve **accurate, robust, and interpretable** metro flow prediction under full, few-shot, and zero-shot settings.  

---

## ✨ Key Features  

- **Spatio-Temporal Encoder**: Attention-based encoder that jointly captures spatial and temporal dependencies in metro networks.  
- **Representation Alignment Module**: Transforms structured spatiotemporal features into token sequences compatible with LLMs.  
- **LLM-Enhanced Prediction**: Leverages reasoning and generalization capabilities of pre-trained LLMs with minimal task-specific fine-tuning.  
- **Interpretability**: Generates **natural language explanations** of prediction results to improve transparency and decision support.  
- **Strong Generalization**: Robust performance across datasets and unseen conditions.  

---

## 📊 Datasets  

We use two large-scale real-world metro datasets:  

- **HZMetro** 
  - Hangzhou Metro system  
  - Duration: **Jan 1 – Jan 25, 2019 (25 days)**  
  - **80 stations**, aggregated traffic statistics  
  - Time resolution: **15-minute intervals**  
  - Records both **inflow** and **outflow** of passengers  

- **SHMetro**  
  - Shanghai Metro system  
  - Duration: **Jul 1 – Sep 30, 2016**  
  - **288 stations**, high spatiotemporal resolution  
  - Time resolution: **15-minute intervals**  
  - Provides fine-grained passenger flow dynamics
## Dataset Download
Baidu Netdisk: [https://pan.baidu.com/s/1lesAk4WOfBQtg0a0XgDfvA](https://pan.baidu.com/s/1lesAk4WOfBQtg0a0XgDfvA)  
Extraction code: **np5p**

For more implementation details, refer to run_MFP-LLM.py.

Installation

Requirements

torch==2.2.2

accelerate==0.28.0

matplotlib==3.7.0

numpy==1.23.5

pandas==1.5.3

scikit_learn==1.2.2

tqdm==4.65.0

transformers==4.31.0

deepspeed==0.14.0

## 🚀 Quick Start  

###  Clone the repo  
```bash
git clone https://github.com/your-username/MFP-LLM.git
cd MFP-LLM

