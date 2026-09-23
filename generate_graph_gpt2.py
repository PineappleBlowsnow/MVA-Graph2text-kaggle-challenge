import argparse
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from transformers import AutoTokenizer

from train_graph_gpt2 import MolGNN, GraphGPT


class TestGraphDataset(Dataset):
    def __init__(self, graph_path: str):
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Test graphs not found: {graph_path}")
        with open(graph_path, "rb") as f:
            self.graphs = pickle.load(f)
        self.ids = [getattr(g, "id", None) for g in self.graphs]
        if any(i is None for i in self.ids):
            raise ValueError("Some graphs are missing the 'id' attribute.")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        g = self.graphs[idx]
        return g, getattr(g, "id")


def collate_graphs_with_ids(batch) -> Tuple[Batch, List[str]]:
    graphs, ids = zip(*batch)
    batch_graph = Batch.from_data_list(list(graphs))
    ids = [str(i) for i in ids]
    return batch_graph, ids


def _infer_gnn_dim_from_checkpoint(state_dict: dict) -> int:
    # projector.0 is the first Linear(gnn_dim -> gpt_dim) in GraphGPT
    w = state_dict.get("projector.0.weight")
    if w is None:
        raise KeyError(
            "Checkpoint is missing 'projector.0.weight' (not a GraphGPT checkpoint?)"
        )
    if not hasattr(w, "shape") or len(w.shape) != 2:
        raise ValueError("Unexpected projector.0.weight shape")
    return int(w.shape[1])


@torch.no_grad()
def generate_descriptions(
    model: GraphGPT,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    max_new_tokens: int,
) -> pd.DataFrame:
    model.eval()

    results = []

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    for batch_graph, batch_ids in loader:
        batch_graph = batch_graph.to(device)

        graph_emb = model.gnn(batch_graph)
        graph_token = model.projector(graph_emb).unsqueeze(1)  # [B,1,emb]

        # Provide a 1-token textual prompt (EOS) so generation has at least one token position after the graph token.
        prompt_ids = torch.full(
            (graph_token.size(0), 1), eos_id, dtype=torch.long, device=device
        )
        prompt_emb = model.gpt2.transformer.wte(prompt_ids)

        inputs_embeds = torch.cat([graph_token, prompt_emb], dim=1)
        attention_mask = torch.ones(
            (inputs_embeds.size(0), inputs_embeds.size(1)),
            dtype=torch.long,
            device=device,
        )

        try:
            out_ids = model.gpt2.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=eos_id,
                eos_token_id=eos_id,
            )
            texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        except Exception:
            # Fallback: greedy decoding with a cached prefix (graph token)
            prefix_out = model.gpt2(inputs_embeds=graph_token, use_cache=True)
            past = prefix_out.past_key_values
            cur_ids = prompt_ids
            generated = []

            finished = torch.zeros((cur_ids.size(0),), dtype=torch.bool, device=device)
            for _ in range(max_new_tokens):
                out = model.gpt2(input_ids=cur_ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1)

                generated.append(next_token)
                finished = finished | (next_token == eos_id)

                cur_ids = next_token.unsqueeze(1)
                if bool(finished.all()):
                    break

            if generated:
                gen_ids = torch.stack(generated, dim=1)
            else:
                gen_ids = torch.empty((cur_ids.size(0), 0), dtype=torch.long, device=device)

            texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for sample_id, text in zip(batch_ids, texts):
            text = (text or "").strip()
            results.append({"ID": sample_id, "description": text})

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate molecular descriptions for test graphs using a trained GraphGPT model."
    )
    parser.add_argument(
        "--model_path", default="best_model.pth", help="Path to trained GraphGPT checkpoint (.pth)"
    )
    parser.add_argument(
        "--test_graphs", default="data/test_graphs.pkl", help="Path to test graphs .pkl"
    )
    parser.add_argument(
        "--output_csv",
        default="kaggle_submission.csv",
        help="Output CSV path (columns: ID, description)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for generation"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=96, help="Max tokens to generate per sample"
    )
    parser.add_argument(
        "--gpt_model_name", default="distilgpt2", help="GPT-2 family model name"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    print(f"Loading checkpoint: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location="cpu")

    gnn_dim = _infer_gnn_dim_from_checkpoint(state_dict)
    print(f"Inferred gnn_dim from checkpoint: {gnn_dim}")

    tokenizer = AutoTokenizer.from_pretrained(args.gpt_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gnn = MolGNN(out_dim=gnn_dim)
    model = GraphGPT(gnn, gpt_model_name=args.gpt_model_name, freeze_gnn=False)
    model.load_state_dict(state_dict)
    model.to(device)

    ds = TestGraphDataset(args.test_graphs)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graphs_with_ids,
        num_workers=0,
    )

    df = generate_descriptions(
        model=model,
        tokenizer=tokenizer,
        loader=dl,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    df.to_csv(args.output_csv, index=False)
    print("=" * 80)
    print(f"Saved {len(df)} rows to: {args.output_csv}")
    print("CSV columns:", list(df.columns))


if __name__ == "__main__":
    main()
