import torch
import matplotlib.pyplot as plt
import argparse
import AdaIN_net as net
from torch.utils.data import DataLoader
import torch.nn as nn
import datetime
from torchvision import transforms
import custom_dataset


#   use a functional tool in the Pytorch that we can crop or resize images  #
def train_transfrom():
    transform_list = [transforms.Resize(size=(512, 512)),
                      transforms.RandomCrop(256),
                      transforms.ToTensor()]
    return transforms.Compose(transform_list)

#   adjust learning rate, since the beginning of the rate cant be too large,
#   the trajectory should be increased from 0.01 to 1
#   and then decrease, the part of decrease more like the graph of cosine.
def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(n_epochs, batch_size, optimizer, net, device):
    print('training ...')
    net.train()
    net.to(device=device)
    losses_train = []
    losses_content = []
    losses_style = []

    for epoch in range(1, n_epochs + 1):
        print('epoch', epoch)
        loss_train_epoch = 0.0
        loss_content_epoch = 0.0
        loss_style_epoch = 0.0
        content_loader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True)
        style_loader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True)
        adjust_learning_rate(optimizer, iteration_count=args.n_epochs)

        for batch in range(len(content_dataset) // args.batch_size):
            print('batch:', batch)

            content_images = next(iter(content_loader)).to(device)
            style_images = next(iter(style_loader)).to(device)

            loss_content, loss_style = net(content_images, style_images)

            loss_train = loss_content + args.gamma * loss_style

            loss_content_epoch += (args.gamma * loss_content).item()
            loss_style_epoch += loss_style.item()
            loss_train_epoch += (args.gamma*loss_content).item()+loss_style.item()



            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()


        losses_train.append(loss_train_epoch / len(content_dataset) // args.batch_size)
        losses_content.append(loss_content_epoch / len(content_dataset) // args.batch_size)
        losses_style.append(loss_style_epoch / len(content_dataset) // args.batch_size)

        print('{} Epoch {}, Combination loss {}'.format(datetime.datetime.now(), epoch,loss_train_epoch / len(content_dataset) // args.batch_size))
        print('{} Epoch {}, Content loss {}'.format(datetime.datetime.now(), epoch, loss_content_epoch / len(content_dataset) // args.batch_size))
        print('{} Epoch {}, Style loss {}'.format(datetime.datetime.now(), epoch, loss_style_epoch / len(content_dataset) // args.batch_size))


    torch.save(net.decoder.state_dict(), args.s)
    print(f"Saved trained model to {args.s}")

    # Save the training loss plot
    plt.plot(range(args.n_epochs), losses_train, label='Content Loss + Style Loss')
    plt.plot(range(args.n_epochs), losses_content, label='Content Loss')
    plt.plot(range(args.n_epochs), losses_style, label='Style Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(args.plot_file, format='png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-content_dir', type=str, default='./COCO100/', help='Directory of content images')
    parser.add_argument('-style_dir', type=str, default='./wikiart100/', help='Directory of style images')
    parser.add_argument('-gamma', type=float, default=1.0, help='Learning rate scheduler gamma')
    parser.add_argument('-e', '--n_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-l', type=str, default='./encoder.pth', help='Path to save the encoder model')
    parser.add_argument('-s', type=str, default='decoder.pth', help='Path to save the decoder model')
    parser.add_argument('-p', type=str, default='./decoder.png', help='Path to save the plot')
    parser.add_argument('-vgg', type=str, default='./encoder.pth')
    parser.add_argument('-cuda', type=str, help='[Y/N]')
    parser.add_argument('--style_weight', type=float, default=0.01)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-lr_decay', type=float, default=5e-5)
    parser.add_argument('--plot_file', type=str, default='loss.png')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    decoder = net.encoder_decoder.decoder
    vgg = net.encoder_decoder.encoder

    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    net = net.AdaIN_net(vgg, decoder)

    content_tf = train_transfrom()
    style_tf = train_transfrom()
    optimizer = torch.optim.Adam(net.decoder.parameters(), lr=args.lr)

    content_dataset = custom_dataset.custom_dataset(args.content_dir, content_tf)
    style_dataset = custom_dataset.custom_dataset(args.style_dir, style_tf)

    train(args.n_epochs, args.batch_size, optimizer, net, device)
