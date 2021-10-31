import click
import argparse
import math
import random
import os
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.nn.functional import smooth_l1_loss
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from model import Generator, Discriminator, ConvLayer, Upsample, Downsample
from train import get_data_loader, parse_args, sample_data, requires_grad, d_logistic_loss, g_nonsaturating_loss, \
    d_r1_loss
from distributed import get_rank
from vqvae import VQVAE


def main():
    device = 'cuda'
    args = parse_args()

    loader = get_data_loader(args)
    data_gen = sample_data(loader)
    generator = UNet().to(device)

    vqvae = VQVAE().to(device)
    vqvae.load_state_dict(torch.load(args.vqvae))
    requires_grad(vqvae, False)

    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)

    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    if args.ckpt is not None:
        print("loading model from checkpoint:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        make_para = lambda model: nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        generator = make_para(generator)
        discriminator = make_para(discriminator)

        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    pbar = range(args.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    r1_loss, enc_loss = 0.0, 0.0
    for idx in pbar:
        i = idx + args.start_iter
        if i > args.iter:
            print("Done!")
            break

        # First step, train discriminator
        images1 = next(data_gen).to(device)
        images2 = next(data_gen).to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        g_input = torch.cat((images1, images2), dim=1)
        fake_img = generator(g_input)
        real_img = next(data_gen).to(device)

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_loss = float(d_loss.item())

        # R1 normalization every so many iterations
        if i % args.d_reg_every == 0:
            real_img = next(data_gen).to(device)
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            r1_loss = (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0])
            r1_loss.backward()
            d_optim.step()
            r1_loss = float(r1_loss.item())

        # Next Step: Train generator
        images1 = next(data_gen).to(device)
        images2 = next(data_gen).to(device)

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        g_input = torch.cat((images1, images2), dim=1)
        fake_img = generator(g_input)

        if i % args.sample_iters == 0:
            utils.save_image(
                torch.cat((images1, images2, fake_img), dim=0),
                f"sample/{str(i).zfill(6)}.png",
                nrow=3,
                normalize=True,
                range=(-1, 1),
            )

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()
        g_loss = float(g_loss.item())

        # Next Step: Encoder loss on generated images
        if i % args.g_reg_every:
            images1 = next(data_gen).to(device)
            images2 = next(data_gen).to(device)

            g_input = torch.cat((images1, images2), dim=1)
            fake_img = generator(g_input)
            zt1, zb1 = vqvae.encode(fake_img)[0:2]
            zt2, zb2 = vqvae.encode(images1)[0:2]
            zt3, zb3 = vqvae.encode(images2)[0:2]
            enc_loss = -args.encoder_regularize * (
                    smooth_l1_loss(zt1, zt2) + smooth_l1_loss(zb1, zb2) +
                    smooth_l1_loss(zt1, zt3) + smooth_l1_loss(zb1, zb3))
            generator.zero_grad()
            enc_loss.backward()
            g_optim.step()
            enc_loss = float(enc_loss.item())

        if get_rank() == 0:
            pbar.set_description(f"d: {d_loss:.4f}; g: {g_loss:.4f}; enc: {enc_loss:.4f}; r1: {r1_loss:.4f}")

        if i % args.checkpoint_iters == 0:
            torch.save(
                {
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "args": args,
                },
                f"checkpoint/{str(i).zfill(6)}.pt",
            )


class UNet(nn.Module):
    def __init__(self, size=256, channel_multiplier=2):
        super().__init__()
        channels = {
            4: 512, 8: 512, 16: 512, 32: 512, 64: 256,
            128: 128, 256: 64, 512: 32, 1024: 16,
        }

        self.input = ConvLayer(6, channels[size], 1)

        self.resblocks_down = nn.ModuleList()
        self.resblocks_up = nn.ModuleList()

        s = size
        while s > 16:
            c1, c2 = channels[s], channels[s//2]
            self.resblocks_down.append(ResBlockDown(c1, c2))
            self.resblocks_up.insert(0, ResBlockUp(c2, c1))
            s = s // 2

        self.output = ConvLayer(channels[size], 3, 1)

    def forward(self, x):
        x = self.input(x)

        skips = []

        for resblock in self.resblocks_down:
            x = resblock(x)
            skips.append(x)

        for resblock in self.resblocks_up:
            x = resblock(x + skips.pop())

        x = self.output(x)
        return x


class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.resblock = ResBlock(in_channel, out_channel)
        self.downsample = nn.AvgPool2d((2, 2), stride=2)

    def forward(self, x):
        x = self.resblock(x)
        x = self.downsample(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3)
        self.skip = ConvLayer(in_channel, out_channel, 1, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        return out


class ResBlockUp(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.resblock = ResBlock(in_channel, out_channel)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = self.upsample(x)
        x = self.resblock(x)
        return x


if __name__ == '__main__':
    main()