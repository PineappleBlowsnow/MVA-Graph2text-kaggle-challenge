# Local challenge data

Obtain the ALTeGraD challenge files through the authorized course/competition channel, then place `train_graphs.pkl`, `validation_graphs.pkl` and `test_graphs.pkl` here. They are not redistributed by this repository.

The source expects a pickled list of PyTorch Geometric graph objects, with `x` (nodes × 9 categorical atom features), `edge_index` (2 × edges), `edge_attr` (edges × 3 categorical bond features), and `id`. Training and validation also need `description`. IDs must match the string keys in the embedding CSVs. `generate_description_embeddings.py` writes `train_embeddings.csv` and `validation_embeddings.csv` locally.

This contract was inferred from the implementation; no pickle file was opened during packaging. Only deserialize files from a trusted source. Do not commit course data, embeddings, generated descriptions or checkpoints.
