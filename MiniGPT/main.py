import math
import os
import urllib.request
import argparse
import csv

import torch
import torch.nn as nn
import torch.nn.functional as f

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CKPT = "minigpt.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Model ---------------- #
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = (B, T, self.n_head, C // self.n_head)
        q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(f.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mpl = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mpl(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_embd=192, n_head=6, n_layer=6, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tom_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tom_emb(idx) + self.pos_emb(pos)
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = f.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(f.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ---------------- Data ---------------- #
def load_text(path):
    if not os.path.exists(path):
        print(f"Downloading sample dataset to {path}......")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad
def estimate_loss(model, splits, block_size, batch_size, iters=20):
    model.eval()
    out = {}
    for name, d in splits.items():
        losses = [model(*get_batch(d, block_size, batch_size))[1].item() for _ in range(iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


# ---------------- Commands ---------------- #
def train(args):
    text = load_text(args.data)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    cfg = dict(vocab_size=len(chars), block_size=args.block_size, n_embd=args.n_embd,
               n_head=args.n_head, n_layer=args.n_layer, dropout=0.1)
    model = MiniGPT(**cfg).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters on {device}")

    history = []
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(args.steps + 1):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))  # cosine decay
        for g in opt.param_groups:
            g["lr"] = lr
        if step % 250 == 0:
            l = estimate_loss(model, splits, args.block_size, args.batch_size)
            print(f"step {step:5d} | train {l['train']:.3f} | val {l['val']:.3f}")
            history.append((step, l["train"], l["val"]))

        with open("loss_log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["steps","train", "val"])
            w.writerows(history)

        xb, yb = get_batch(splits["train"], args.block_size, args.batch_size)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        opt.step()
    torch.save({"model": model.state_dict(),"cfg": cfg, "chars": chars}, CKPT)
    print(f"Saved {CKPT}")

def generate(args):
    ck = torch.load(CKPT, map_location=device)
    chars = ck["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    model = MiniGPT(**ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    ids = [stoi[c] for c in args.prompt if c in stoi] or [0]
    idx = torch.tensor([ids], device=device)
    out = model.generate(idx, args.tokens, args.temperature, args.top_k)[0].tolist()
    print("".join(chars[i] for i in out))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", default="input.txt")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch_size", type=int, default=64)
    t.add_argument("--block_size", type=int, default=128)
    t.add_argument("--n_embd", type=int, default=192)
    t.add_argument("--n_head", type=int, default=6)
    t.add_argument("--n_layer", type=int, default=6)
    t.add_argument("--lr", type=float, default=3e-4)
    t.set_defaults(fn=train)

    g = sub.add_parser("generate")
    g.add_argument("--prompt", default="\n")
    g.add_argument("--tokens", type=int, default=300)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top_k", type=int, default=40)
    g.set_defaults(fn=generate)

    a = p.parse_args()
    a.fn(a)
