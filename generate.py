import argparse
import os
from os.path import join

from PIL import Image
import torch
from torchvision import utils
from tqdm import tqdm


def generate(args, g_ema, device, mean_latent):
    im_count = 0
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )
            norm_range(sample, (-1, 1))

            ndarr = sample.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            for imarr in ndarr:
                im = Image.fromarray(imarr)
                im_count += 1
                imdir = join(args.outdir, f"{(im_count//1000):03d}")
                os.makedirs(imdir, exist_ok=True)
                impath = join(imdir, f"im{(im_count % 1000):03d}.png")
                im.save(impath)


def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))


def norm_range(t, value_range):
    if value_range is not None:
        norm_ip(t, value_range[0], value_range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )

    parser.add_argument(
        "--nrow",
        type=int,
        default=1,
        help="number of images per row of output image."
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    parser.add_argument(
        "--arch",
        type=str,
        default='swagan',
        choices=["swagan", "stylegan2"],
        help="architecture to use"
    )

    parser.add_argument( "outdir", help="Where to save the images." )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    if args.arch == 'swagan':
        from swagan import Generator
    else:
        from model import Generator

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
