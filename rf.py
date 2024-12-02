# implementation of Rectified Flow for simple minded people like me.
import argparse

import torch
from muon import Muon
import heavyball

class RF:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        # convert ot bf16
        zt, t, cond = zt.bfloat16(), t.bfloat16(), cond.long()
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)
            z, t, cond = z.bfloat16(), t.bfloat16(), cond.long()
            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images

adam_beta_options = {
    "0.9/0.95": (0.90, 0.95),
    "0.95/0.99": (0.95, 0.99),
    "0.95/0.95": (0.95, 0.95),
}
def muon_optimizer(model, lr=0.01, lr2=5e-4, momentum=0.95, beta_option="0.9/0.95"):
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        # embed, unembed will always go to adamw, if ndim < 2 goes to adamw, else muon
        if ("init_conv_seq" in name) or ("final_layer" in name) or (param.ndim < 2):
            adamw_params.append(param)  
            # print("ADAmw", name)
        else: 
            muon_params.append(param)
            # print("Muon", name)

    
    beta_opts = adam_beta_options[beta_option]
    optimizer = Muon(muon_params, lr=lr, momentum=momentum, 
                    adamw_params=adamw_params, adamw_lr=lr2, adamw_betas=beta_opts, adamw_wd=0.0)
    return optimizer

def adam_optimizer(model, lr=5e-4, beta_option="0.9/0.95"):
    beta_opts = adam_beta_options[beta_option]
    return torch.optim.Adam(model.parameters(), lr=lr, betas=beta_opts)

def psgd_optimizer(model, lr=5e-4, b1=0.9): #psgd only has one beta
    return heavyball.PSGDKron(model.parameters(), lr=lr, beta=b1)

def soap_optimizer(model, lr=5e-4, beta_option="0.9/0.95"):
    (beta, sbeta) = adam_beta_options[beta_option]
    return heavyball.PrecondSchedulePaLMSOAP(model.parameters(), lr=lr, beta=beta, shampoo_beta=sbeta)

@torch.no_grad()
def calculate_val_loss(model, rf, val_dataloader):
    model.eval()
    total_loss = 0
    val_lossbin = {i: 0 for i in range(10)}
    val_losscnt = {i: 1e-6 for i in range(10)}
    
    for x, c in val_dataloader:
        x, c = x.cuda(), c.cuda()
        loss, blsct = rf.forward(x, c)
        total_loss += loss.item()
        
        # count based on t
        for t, l in blsct:
            val_lossbin[int(t * 10)] += l
            val_losscnt[int(t * 10)] += 1
    
    avg_loss = total_loss / len(val_dataloader)
    model.train()
    return avg_loss, val_lossbin, val_losscnt

def sample(rf, run_name):
    rf.model.eval()
    with torch.no_grad():
        for cd_id in [3, 4, 6]:
            cond = torch.zeros(16).cuda() + cd_id
            uncond = torch.ones_like(cond) * 10
            # generator ith seed of 0
            init_noise = torch.randn(16, channels, 32, 32, generator= torch.Generator().manual_seed(0) ).cuda()
            images = rf.sample(init_noise, cond, uncond)
            # image sequences to gif
            gif = []
            for image in images:
                # unnormalize
                image = image * 0.5 + 0.5
                image = image.clamp(0, 1)
                x_as_image = make_grid(image.float(), nrow=4)
                img = x_as_image.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                gif.append(Image.fromarray(img))

            gif[0].save(
                f"contents/sample_{run_name}_{epoch}.gif",
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )

            # wandb log
            wandb.log({
                f"sample_{cd_id}": wandb.Image(f"contents/sample_{run_name}_{epoch}.gif"),
                f"sample_{cd_id}_last": wandb.Image(f"contents/sample_{run_name}_{epoch}_last.png")
                       })
            

            # last_img = gif[-1]
            # last_img.save(f"contents/sample_{run_name}_{epoch}_last.png")

    rf.model.train()

if __name__ == "__main__":
    # train class conditional RF on mnist.
    # set seed to 12
    torch.manual_seed(12)
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    from tqdm import tqdm

    import wandb
    from dit import DiT_Llama

    parser = argparse.ArgumentParser(description="use cifar?")
    parser.add_argument("--cifar", action="store_true")
    # muon lr , adam lr , momentum hps
    parser.add_argument("--muon_lr", type=float, default=0.01)
    parser.add_argument("--adam_lr", type=float, default=5e-4)
    parser.add_argument("--psgd_lr", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--optim", type=str, default="adam")
    # beta options for adam ya know
    parser.add_argument("--beta_option", type=str, default="0.9/0.95")
    # batch size
    parser.add_argument("--batch_size", type=int, default=512)

    args = parser.parse_args()
    CIFAR = args.cifar
    MLR, ALR, MOMENTUM, OPTIM, BETA_OPTS, BATCH_SIZE, PSGD_LR = args.muon_lr, args.adam_lr, args.momentum, args.optim, args.beta_option, args.batch_size, args.psgd_lr


    if CIFAR:
        dataset_name = "cifar"
        fdatasets = datasets.CIFAR10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 3
        model = DiT_Llama(
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10
        ).cuda()

    else:
        dataset_name = "mnist"
        fdatasets = datasets.MNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 1
        model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10
        ).cuda()

    # model compile
    # model = torch.compile(model)
    # print("Model has been compiled")

    # convert to bf16
    model = model.bfloat16()
    model = torch.compile(model)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = RF(model)
    # optimizer = optim.Adam(model.parameters(), lr=5e-4)
    if OPTIM == "adam":
        optimizer = adam_optimizer(model, lr=ALR, beta_option=BETA_OPTS)
    elif OPTIM == "muon":
        optimizer = muon_optimizer(model, lr=MLR, lr2=ALR, momentum=MOMENTUM, beta_option=BETA_OPTS)
    elif OPTIM == "psgd":
        optimizer = psgd_optimizer(model, lr=PSGD_LR, b1=MOMENTUM)
    elif OPTIM == "soap":
        optimizer = soap_optimizer(model, lr=ALR, beta_option=BETA_OPTS)


    
    # optimizer = muon_optimizer(model)
    criterion = torch.nn.MSELoss()

    train_dataset = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataset = fdatasets(root="./data", train=False, download=True, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False, drop_last=True)
    add_hp = ""#f"{MLR}/{ALR}_{MOMENTUM}" if OPTIM == "muon" else (f"{ALR}" if OPTIM == "adam" else f"{PSGD_LR}_{MOMENTUM}")
    if OPTIM == "muon":
        add_hp += f"{MLR}/{ALR}_{MOMENTUM}"
    elif OPTIM == "adam":
        add_hp += f"{ALR}"
    elif OPTIM == "psgd":
        add_hp += f"{PSGD_LR}_{MOMENTUM}"
    elif OPTIM == "soap":
        add_hp += f"{ALR}"

    run_name = f"{OPTIM}_{add_hp}_{BETA_OPTS}_{BATCH_SIZE}"
    run_cfg = {
        "muon_lr": MLR,
        "adam_lr": ALR,
        "optim": OPTIM,
        "momentum": MOMENTUM,
        "beta_opts": BETA_OPTS,
        "batch_size": BATCH_SIZE,
        "psgd_lr": PSGD_LR
    }

    wandb.init(project=f"rf_{dataset_name}", name=run_name, config=run_cfg)

    n_sample = 0
    global_step = 0
    for epoch in range(50):
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}

        losses = []
        
        for i, (x, c) in enumerate(tqdm((dataloader))):
            x, c = x.cuda(), c.cuda()
            # convert x&c to bf16
            # x, c = x.bfloat16(), c.bfloat16()
            optimizer.zero_grad()
            loss, blsct = rf.forward(x, c)
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item(), "n_sample":n_sample, "epoch": epoch}, step=global_step)

            # count based on t
            for t, l in blsct:
                lossbin[int(t * 10)] += l
                losscnt[int(t * 10)] += 1

            # # every 20 steps, we calculate val loss
            # if i % 50 == 0:
            #     
            #     # print(f"Epoch: {epoch}, {i} steps, val loss: {val_loss}")
            #     wandb.log({"val_loss": val_loss, "n_sample":n_sample, "epoch": epoch}, step=global_step)
            n_sample += x.size(0)
            global_step += 1
            losses.append(loss.item())

        # log
        for i in range(10):
            print(f"Epoch: {epoch}, {i} range loss: {lossbin[i] / losscnt[i]}")
        bin_losses = {f"lossbin_{i}": lossbin[i] / losscnt[i] for i in range(10)}
        bin_losses["n_sample"] = n_sample
        bin_losses["epoch"] = epoch
        val_loss, val_lossbin, val_losscnt = calculate_val_loss(model, rf, val_dataloader)

        # val loss log
        bin_losses["epoch_val_loss"] = val_loss
        # mean epoch loss
        bin_losses["epoch_loss"] = np.mean(losses)
        
        wandb.log(bin_losses, step=global_step)

        # sample(rf, run_name)
        # print("sample done")

        

