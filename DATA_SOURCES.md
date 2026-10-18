# Data Sources and Boundaries

## FNSPID

The primary historical source is **FNSPID: A Comprehensive Financial News Dataset in Time Series** by Zihan Dong, Xinyu Fan and Zhiyuan Peng.

- Project: https://github.com/Zdong104/FNSPID_Financial_News_Dataset
- Paper: https://arxiv.org/abs/2402.06698
- Dataset host: https://huggingface.co/datasets/Zihan1004/FNSPID
- Dataset license: CC BY-NC 4.0

Suggested citation:

```bibtex
@misc{dong2024fnspid,
  title={FNSPID: A Comprehensive Financial News Dataset in Time Series},
  author={Zihan Dong and Xinyu Fan and Zhiyuan Peng},
  year={2024},
  eprint={2402.06698},
  archivePrefix={arXiv},
  primaryClass={q-fin.ST}
}
```

## SEC supplement

The optional collection script retrieves a small set of public SEC filing metadata and documents from SEC endpoints. It identifies the client through a configurable user-agent and preserves accession numbers and retrieval metadata.

## Repository boundary

This public repository intentionally excludes:

- full news article text;
- downloaded raw datasets;
- publisher HTML;
- local annotation workbooks;
- credentials and API keys.

Only project code, documentation, aggregate metrics and non-identifying figures are included. Users must obtain third-party datasets from the original hosts and comply with their terms.
