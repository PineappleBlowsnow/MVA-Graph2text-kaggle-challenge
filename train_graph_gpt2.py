import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import os

from data_utils import MoleculeGenDataset, GPTCollate

from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, GINEConv, global_add_pool


class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.emb_atomic_num = nn.Embedding(119, hidden_dim)
        self.emb_chirality = nn.Embedding(9, hidden_dim)
        self.emb_degree = nn.Embedding(11, hidden_dim)
        self.emb_charge = nn.Embedding(12, hidden_dim)
        self.emb_num_hs = nn.Embedding(9, hidden_dim)
        self.emb_rad_el = nn.Embedding(5, hidden_dim)
        self.emb_hybrid = nn.Embedding(8, hidden_dim)
        self.emb_aromatic = nn.Embedding(2, hidden_dim)
        self.emb_ring = nn.Embedding(2, hidden_dim)

    def forward(self, x):
        x = x.long()
        atomic_num = self.emb_atomic_num(x[:, 0].clamp(0, 118))
        chirality = self.emb_chirality(x[:, 1].clamp(0, 8))
        degree = self.emb_degree(x[:, 2].clamp(0, 10))
        charge = self.emb_charge(x[:, 3].clamp(0, 11))
        num_hs = self.emb_num_hs(x[:, 4].clamp(0, 8))
        rad_el = self.emb_rad_el(x[:, 5].clamp(0, 4))
        hybrid = self.emb_hybrid(x[:, 6].clamp(0, 7))
        aromatic = self.emb_aromatic(x[:, 7].clamp(0, 1))
        ring = self.emb_ring(x[:, 8].clamp(0, 1))

        h = (atomic_num + chirality + degree + charge +
             num_hs + rad_el + hybrid + aromatic + ring)
        return h


class EdgeEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.emb_bond_type = nn.Embedding(23, hidden_dim)
        self.emb_stereo = nn.Embedding(6,hidden_dim)
        self.emb_conjugated = nn.Embedding(2,hidden_dim)

    def forward(self, edge_attr):
        return self.emb_bond_type(edge_attr[:, 0]) + \
               self.emb_stereo(edge_attr[:, 1]) + \
               self.emb_conjugated(edge_attr[:, 2])


class MolGNN(nn.Module):
    def __init__(self, hidden=128, out_dim=256, layers=3):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden)
        self.edge_encoder = EdgeEncoder(hidden)

        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
            self.convs.append(GINEConv(mlp))

        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, batch: Batch):
        h = self.atom_encoder(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)

        for conv in self.convs:
            h = conv(h, batch.edge_index, edge_attr=edge_attr)
            h = F.relu(h)

        g = global_add_pool(h, batch.batch)
        g = self.proj(g)
        g = F.normalize(g, dim=-1)
        return g


class GraphGPT(nn.Module):
    def __init__(self, gnn_encoder, gpt_model_name='distilgpt2', freeze_gnn=True):
        super().__init__()
        self.gnn = gnn_encoder

        if freeze_gnn:
            for param in self.gnn.parameters():
                param.requires_grad = False

        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt_model_name)

        gpt_dim = self.gpt2.config.n_embd
        # Infer the actual output dimension of the provided GNN encoder.
        # For our MolGNN, it's the out_features of the last Linear in self.proj.
        if hasattr(self.gnn, "proj") and isinstance(self.gnn.proj, nn.Sequential):
            last = self.gnn.proj[-1]
            if isinstance(last, nn.Linear):
                gnn_dim = last.out_features
            else:
                raise ValueError("Unsupported gnn.proj format: expected last layer to be nn.Linear")
        else:
            raise ValueError("Unsupported GNN encoder: expected a .proj nn.Sequential")

        self.projector = nn.Sequential(
            nn.Linear(gnn_dim, gpt_dim),
            nn.ReLU(),
            nn.Linear(gpt_dim, gpt_dim)
        )

    def forward(self, batch, text_input_ids, text_attention_mask):
        graph_emb = self.gnn(batch)
        graph_token = self.projector(graph_emb).unsqueeze(1)

        wte = self.gpt2.transformer.wte
        text_emb = wte(text_input_ids)

        inputs_embeds = torch.cat([graph_token, text_emb], dim=1)

        device = text_attention_mask.device
        batch_size = text_attention_mask.size(0)
        ones = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        extended_attention_mask = torch.cat([ones, text_attention_mask], dim=1)

        ignore_labels = torch.full((batch_size, 1), -100, dtype=torch.long, device=device)
        labels = torch.cat([ignore_labels, text_input_ids], dim=1)
        labels = labels.masked_fill(extended_attention_mask == 0, -100)

        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=labels
        )
        return outputs.loss


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Training")

    for batch_graph, input_ids, attention_mask in loop:
        batch_graph = batch_graph.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()
        loss = model(batch_graph, input_ids, attention_mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        loop = tqdm(loader, desc="Validating")
        for batch_graph, input_ids, attention_mask in loop:
            batch_graph = batch_graph.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            loss = model(batch_graph, input_ids, attention_mask)
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)


def main():
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 10

    TRAIN_GRAPHS = 'data/train_graphs.pkl'
    VAL_GRAPHS = 'data/validation_graphs.pkl'
    GNN_WEIGHTS = 'model_checkpoint.pt'
    SAVE_DIR = 'checkpoints_gen'
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(TRAIN_GRAPHS):
        print(f"❌ Error: {TRAIN_GRAPHS} not found!")
        return

    gpt_collate = GPTCollate(tokenizer='distilgpt2', max_len=128)

    train_dataset = MoleculeGenDataset(graph_path=TRAIN_GRAPHS)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=gpt_collate,
        num_workers=0
    )

    val_loader = None
    if os.path.exists(VAL_GRAPHS):
        val_dataset = MoleculeGenDataset(graph_path=VAL_GRAPHS)
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=gpt_collate,
            num_workers=0
        )
        print(f"✅ Loaded Validation Set: {len(val_dataset)} samples")
    else:
        print("Warning: Validation graphs not found, skipping validation.")

    gnn_encoder = MolGNN(out_dim=768)

    if not os.path.exists(GNN_WEIGHTS):
        raise FileNotFoundError(
            f"{GNN_WEIGHTS} is required for the frozen encoder; run train_retrieval.py first."
        )
    checkpoint = torch.load(GNN_WEIGHTS, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {key.removeprefix('module.'): value for key, value in state_dict.items()}
    # Fail on incompatible weights rather than freezing an untrained encoder.
    gnn_encoder.load_state_dict(state_dict, strict=True)
    print("Pretrained GNN weights loaded successfully")

    model = GraphGPT(gnn_encoder, gpt_model_name='distilgpt2', freeze_gnn=True)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR)

    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"   Train Loss: {train_loss:.4f}")

        if val_loader:
            val_loss = evaluate(model, val_loader, device)
            print(f"   Val Loss:   {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(SAVE_DIR, "best_model.pth")
                torch.save(model.state_dict(), best_path)
                print(f"   New best model saved")

        last_path = os.path.join(SAVE_DIR, "last_model.pth")
        torch.save(model.state_dict(), last_path)
        print(f"   Checkpoint saved to {last_path}")

if __name__ == "__main__":
    main()
