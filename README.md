# High-dimensional Asymptotics of VAEs:Threshold of Posterior Collapse and Dataset-Size Dependence of Rate-Distortion Curve

## Abstract
In variational autoencoders (VAEs), the variational posterior often collapses to the prior, known as posterior collapse, which leads to poor representation learning quality.
An adjustable hyperparameter beta has been introduced in VAEs to address this issue.
This study sharply evaluates the conditions under which the posterior collapse occurs with respect to beta and dataset size by analyzing a minimal VAE in a high-dimensional limit. Additionally, this setting enables the evaluation of the rate-distortion curve of the VAE.
Our results show that, unlike typical regularization parameters, VAEs face "inevitable posterior collapse" beyond a certain beta threshold, regardless of dataset size.
Moreover, the dataset-size dependence of the derived rate-distortion curve suggests that relatively large datasets are required to achieve a rate-distortion curve with high rates.
These findings robustly explain generalization behavior observed in various real datasets with highly non-linear VAEs.

---

## Requirements
To install the required dependencies, use:

```bash
pip install -r requirements.txt
```

### **Required Packages & Versions**
- `torch==2.6.0`
- `numpy==2.2.4`
- `matplotlib==3.10.1`
- `seaborn==0.13.2`

---

## License
This project is licensed under the **BSD 3-Clause License**. See [LICENSE](LICENSE.txt) for details.

---

## Citation
If you use this work in your research, please cite:

```bibtex
@misc{
ichikawa2025highdimensional,
title={High-dimensional Asymptotics of {VAE}s: Threshold of Posterior Collapse and Dataset-Size Dependence of Rate-Distortion Curve},
author={Yuma Ichikawa and Koji Hukushima},
year={2025},
url={https://openreview.net/forum?id=BdPbmgJ2jo}
}
```

