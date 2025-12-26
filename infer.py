import yaml, torch
from model.model import CharCodeButterfly
from utils.streaming import ButterflyStreamCache

cfg = yaml.safe_load(open("config/model.yaml"))

device = "cuda"

model = CharCodeButterfly(cfg).to(device).eval()
model.load_state_dict(torch.load("checkpoint.pt", map_location=device))

# streaming cache for butterfly layers
BUTTERFLY_LAYERS = cfg["butterfly_passes"] * cfg["butterfly_layers"]
cache = ButterflyStreamCache(BUTTERFLY_LAYERS, cfg["chunk_size"])

def sample(logits, temperature=0.8):
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1)

def generate(prompt, max_new_tokens=1000):
    x = torch.tensor(list(prompt.encode("latin-1")), device=device).unsqueeze(0)

    # warm-up full forward once
    with torch.no_grad():
        _ = model(x)

    for _ in range(max_new_tokens):
        # Only last token flows through streaming path
        last_token = x[:, -1:]

        # forward_stream updates only O(log N) butterfly blocks
        logits = model.forward_stream(last_token, cache)
        nxt = sample(logits[:, -1])
        x = torch.cat([x, nxt], dim=1)

    return bytes(x[0].tolist()).decode("latin-1")

if __name__ == "__main__":
    print(generate("int main() {"))
