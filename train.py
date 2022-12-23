"""Training procedure for NICE.
"""

import argparse
import torch
import torchvision
from torchvision import transforms
from VAE import Model, Averager
import os
import matplotlib.pyplot as plt


def train(vae, trainloader, optimizer, epoch, device):
    vae.train()  # set to training mode
    avg = Averager()
    for x, _ in trainloader:
        x = x.to(device)
        vae.zero_grad()
        z_mu, z_log_var, reconstruct = vae(x)
        loss = vae.loss(x=x, recon=reconstruct, mu=z_mu, logvar=z_log_var)
        loss.backward()
        optimizer.step()
        avg.add(-loss)
    return avg.val().item()


def test(vae, testloader, filename, epoch, device):
    vae.eval()  # set to inference mode
    with torch.no_grad():
        avg = Averager()
        for x, _ in testloader:
            x = x.to(device)
            z_mu, z_log_var, reconstruct = vae(x)
            loss = vae.loss(x=x, recon=reconstruct, mu=z_mu, logvar=z_log_var)
            avg.add(-loss)
        samples = vae.sample(100).cpu()
        a, b = samples.min(), samples.max()
        samples = (samples - a) / (b - a + 1e-10)
        samples = samples.view(-1, 1, 28, 28)
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + 'epoch%d.png' % epoch)
    return avg.val().item()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1. / 256.)),  # dequantization
        transforms.Normalize((0.,), (257. / 256.,)),  # rescales to [0,1]
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=0)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
               + 'batch%d_' % args.batch_size \
               + 'mid%d_' % args.latent_dim

    vae = Model(latent_dim=args.latent_dim, device=device).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    train_elbos = []
    test_elbos = []
    best_elbo = float('-inf')
    for epoch in range(args.epochs):
        train_elbo = train(vae=vae, trainloader=trainloader, optimizer=optimizer, epoch=epoch, device=device)
        test_elbo = test(vae=vae, testloader=testloader, filename=filename, epoch=epoch, device=device)
        if test_elbo > best_elbo:
            best_elbo = test_elbo
            torch.save(vae.state_dict(), os.path.join('models', filename + str(epoch) + '.pt'))
        train_elbos.append(train_elbo)
        test_elbos.append(test_elbo)
        print(f'Epoch {epoch}, train elbo :{train_elbo}, test elbo {test_elbo}')
    plt.xlabel("epoch")
    plt.ylabel("elbo")
    plt.title("VAE training")
    plt.scatter(list(range(args.epochs)), train_elbos, c='r')
    plt.scatter(list(range(args.epochs)), test_elbos, c='b')
    plt.savefig(f'{filename}.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='fashion-mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)

    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    os.makedirs('models', exist_ok=True)
    main(args)
