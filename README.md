<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/marble_logo.jpg">
    <source media="(prefers-color-scheme: light)" srcset="assets/marble_logo.jpg">
    <img alt="Marble" src="assets/marble_logo.jpg" width="360">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/Marble/Marble/actions/workflows/cmake-build.yml"><img alt="CMake Build" src="https://github.com/MarbleSolver/Marble/actions/workflows/cmake-build.yml/badge.svg"></a>
  <a href="https://roboticexplorationlab.org/Marble/"><img alt="Docs" src="https://img.shields.io/badge/docs-dev-blue.svg"></a>
</p>

A C++ solver for quadratic programs with linear complementarity constraints, with Python and Julia bindings.

## Setup and Installation

Please see the [installation instructions](https://roboticexplorationlab.org/Marble/getting_started/installation/) on the docs website for detailed instructions on building, installing, and using Marble in C++, Julia, and Python.

<details>
<summary><b>Choosing the retraction map</b></summary>

The retraction map is selected at compile time via the `MARBLE_RETRACTION` CMake option (default `SOFTPLUS`). With $k$ the relaxation parameter ($\kappa$) and $x$ the slack:

| `MARBLE_RETRACTION` | Retraction map |
| --- | --- |
| `SOFTPLUS` (default) | $p(x) = \tfrac{1}{2}\left(x + \sqrt{x^2 + 4k}\right)$ |
| `EXP` | $p(x) = \sqrt{k}\,e^{x}$ |
| `EXP_SCALED` | $p(x) = \sqrt{k}\,e^{x/\sqrt{k}}$ |

Set it when configuring, e.g.:

```bash
cmake --preset all -DMARBLE_RETRACTION=EXP_SCALED
```

</details>

## Attribution

If you use this work in your research, please cite it as follows:

```bibtex
@article{bishop2026complementarityconstructionliegroupapproach,
      title={Complementarity by Construction: A Lie-Group Approach to Solving Quadratic Programs with Linear Complementarity Constraints}, 
      author={Arun L. Bishop and Micah I. Reich and Zachary Manchester},
      year={2026},
      eprint={2604.11991},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2604.11991}, 
}
```