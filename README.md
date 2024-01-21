# nwport

**Network portrait** is matrix $B$ constructed from a graph such that $B_{l,k}$ is the number of nodes having $k$ nodes at a distance $l$. It encodes a lot of the network's structural properties.

This repository represents the research work done in proceedings of Idea's 2021 Scientific School: Mathematics, Theoretical Physics and Mathematical Methods of Data Analysis in Neuroscience [(link)](https://brain.scientificideas.org/sirius-school/en).

The work is inspired by [Bagrow & Bollt, 2019](https://doi.org/10.1007/s41109-019-0156-x).

View project's final [presentation](https://docs.google.com/presentation/d/1HKaksL892e7ukNYzRvHkvWK-PFPo3V-RFO6PCoXvQMs/edit?usp=sharing).



## Inside
- Network portrait implementation;
- Support for directed & undirected graphs;
- Network portrait heatmap plots;
- Calculation of KL-divergence & Jensen-Shannon divergence;
- Portraits of model (regular & random) networks;
- Animations of model network portraits across parameter ranges;
- Implementation of attacks on networks (gradual node removal) in different modes.

## Data
- Model networks:
  - [Erdős–Rényi](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model)
  - [Random Regular](https://en.wikipedia.org/wiki/Random_regular_graph)
  - [Barabási–Albert](https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model)
  - [Watts–Strogatz](https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model)
- Human brain connectomes courtesy of the [PIT Bioinformatics group](https://braingraph.org/cms/download-pit-group-connectomes/);
- C. Elegans connectomes courtesy of [Neurodata](https://neurodata.io/project/connectomes/).
