# ARES: Scalable and Practical Gradient Inversion Attack in Federated Learning through Activation Recovery


## 🚀 Abstract
Federated Learning (FL) enables collaborative model training by sharing model updates instead of raw data, aiming to protect user privacy. However, recent studies reveal that these shared updates can inadvertently leak sensitive training data through gradient inversion attacks (GIAs). Among them, active GIAs are particularly powerful, enabling high-fidelity reconstruction of individual samples even under large batch sizes. Nevertheless, existing approaches often require architectural modifications, which limit their practical applicability.
In this work, we bridge this gap by introducing the Activation REcovery via Sparse inversion (ARES) attack, an active GIA designed to reconstruct training samples from large training batches without requiring architectural modifications. Specifically, we formulate the recovery problem as a noisy sparse recovery task and solve it using the generalized Least Absolute Shrinkage and Selection Operator (Lasso). To extend the attack to multi-sample recovery, ARES incorporates the imprint method to disentangle activations, enabling scalable per-sample reconstruction. We further establish the expected recovery rate and derive an upper bound on the reconstruction error, providing theoretical guarantees for the ARES attack.
## 🚩 Getting Started
### Clone the repository
```bash
git clone https://github.com/gongzir1/ARES.git
cd ARES
```

### Environment Setup
```bash
conda create -n ares-env python=3.10 -y
conda activate ares-env
pip install -r requirements.txt
```

### Run Inversion
Launch the provided Jupyter notebook to run the CIFAR-10 example: 
```bash
ARES_run_example_cifar10.ipynb
```
Run all cells to execute the ARES attack pipeline and visualize reconstructed samples.

## 📄 Citation

```bibtex
@inproceedings{gong2026ares,
  title={ARES: Activation Recovery via Sparse Inversion in Federated Learning},
  author={Gong, Zirui and Zhang, Leo Yu and Zhang, Yanjun and Vo, Viet and Zhu, Tianqing and Pan, Shirui and Wang, Cong},
  year={2026},
  booktitle={2026 IEEE Symposium on Security and Privacy (SP)}
}
```
## 🙏 Acknowledgements

We would like to thank the author of **[robbing_the_fed](https://github.com/username/original-repo](https://github.com/lhfowl/robbing_the_fed?tab=readme-ov-file)** for providing the original codebase, which served as a foundation for our work and allowed us to build ARES on top of it.
