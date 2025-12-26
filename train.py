import yaml, torch, os
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.model import CharCodeButterfly
from utils.dataset import CharDataset

cfg = yaml.safe_load(open("config/model.yaml"))

files = ["data/train.txt"]
dataset = CharDataset(files, cfg["max_tokens"])
loader = DataLoader(dataset, batch_size=1)

model = CharCodeButterfly(cfg).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9,0.95), weight_decay=0.1)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

accum = 8
global_step = 0

# Resume if checkpoint exists
if os.path.exists("checkpoint.pt"):
    print("Resuming from checkpoint.pt")
    model.load_state_dict(torch.load("checkpoint.pt"))
    if os.path.exists("optim.pt"):
        opt.load_state_dict(torch.load("optim.pt"))

opt.zero_grad()

for x,y in tqdm(loader):
    x,y = x.cuda(), y.cuda()
    loss = loss_fn(model(x).view(-1,256), y.view(-1))
    loss.backward()
    global_step += 1

    if global_step % accum == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        opt.zero_grad()

    # Save checkpoint every 1000 optimizer steps
    if global_step % (accum * 1000) == 0:
        torch.save(model.state_dict(), "checkpoint.pt")
        torch.save(opt.state_dict(), "optim.pt")
        print("Checkpoint saved")
