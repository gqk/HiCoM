# <div align="center"><b> HiCoM: Hierarchical Coherent Motion for Dynamic Streamable Scenes with 3D Gaussian Splatting</b></div>

<div align="center">

[Qiankun Gao](https://gqk.me)<sup>1, 2</sup>, Jiarui Meng<sup>1</sup>, Chengxiang Wen<sup>1</sup>, Jie Chen<sup>1, 2 ✉️</sup>, [Jian Zhang](https://jianzhang.tech)<sup>1, 3 ✉️</sup>

<sup>1</sup>School of Electronic and Computer Engineering, Peking University
<sup>2</sup>Peng Cheng Laboratory
<sup>3</sup>Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology, <br /> Peking University Shenzhen Graduate School

[[`NeurIPS`](https://neurips.cc/virtual/2024/poster/96081)] [[`arXiv`](https://arxiv.org/pdf/2411.07541)] [[`OpenReview`](https://openreview.net/forum?id=De4VWE4rbz)]

<br />

<img src='https://github.com/user-attachments/assets/8057a6d3-2bfe-4f74-8c7e-4f42d995c09f' width="80%" />

</div>

## 🗞️ News
- [2024/12/19] Code released.
- [2024/10/29] Camera ready submitted.
- [2024/09/26] Accepted to NeurIPS 2024 as poster presentation!

## ⚙️ Experiment

The code is built on [LibGS](https://github.com/Awesome3DGS/LibGS); please familiarize yourself with it before running the experiments.

1. Install dependencies

   ```bash
   pip install .
   ```

2. Run pipeline

   ```
   python main.py --config=config/dynerf.yaml --data.root=<PATH TO SCENE ROOT>
   ```

   The `config` directory contains pre-defined configurations for reproducing the results reported in the paper.

## 📖 Citation

```bibtex
@inproceedings{hicom2024,
  title = {HiCoM: Hierarchical Coherent Motion for Dynamic Streamable Scenes with 3D Gaussian Splatting},
  author={Gao, Qiankun  and Meng, Jiarui and Wen, Chengxiang  and Chen, Jie and Zhang, Jian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2024}
}
```

## 🙌 Acknowledgement

We sincerely thank the authors of [3DGStream](https://github.com/SJoJoK/3DGStream) for their inspiring work and valuable assistance.
We also appreciate the contributions and accessible code provided by related research efforts, including [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [4DGaussians](https://github.com/hustvl/4DGaussians), and [Dynamic3DGS](https://github.com/JonathonLuiten/Dynamic3DGaussians), which have greatly supported our research.
