# ARES: Scalable and Practical Gradient Inversion Attack in Federated Learning through Activation Recovery, 


## 🚀 Abstract
Federated Learning (FL) enables collaborative model training by sharing model updates instead of raw data, aiming to protect user privacy. However, recent studies reveal that these shared updates can inadvertently leak sensitive training data through gradient inversion attacks (GIAs). Among them, active GIAs are particularly powerful, enabling high-fidelity reconstruction of individual samples even under large batch sizes. Nevertheless, existing approaches often require architectural modifications, which limit their practical applicability.
In this work, we bridge this gap by introducing the Activation REcovery via Sparse inversion (ARES) attack, an active GIA designed to reconstruct training samples from large training batches without requiring architectural modifications. Specifically, we formulate the recovery problem as a noisy sparse recovery task and solve it using the generalized Least Absolute Shrinkage and Selection Operator (Lasso). To extend the attack to multi-sample recovery, ARES incorporates the imprint method to disentangle activations, enabling scalable per-sample reconstruction. We further establish the expected recovery rate and derive an upper bound on the reconstruction error, providing theoretical guarantees for the ARES attack.
## 🚩 Getting Started
### 1. Clone the repository
```bash
git clone https://github.com/gongzir1/ARES.git
cd ARES
python -m venv ares-env




📄 Citation 
