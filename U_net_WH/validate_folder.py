import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from utils.data_loading import BasicDataset, CarvanaDataset
from evaluate_folder import evaluate_folder
from unet import UNet
from torchvision import transforms

current_loc = Path(__file__)
dir_val_img = current_loc.parent / 'data/testimgsall/'
dir_val_mask = current_loc.parent / 'data/testmasksall/'

def validate(net,
            device,
            batch_size: int = 1,
            save_checkpoint: bool = True,
            img_scale: float = 0.5,
            amp: bool = False,
            cropsize: int = 256,
            logimage: bool = False):

    # 1. Create dataset
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    val_set = BasicDataset(dir_val_img, dir_val_mask, img_scale, transform=transform, cropsize=None)

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)

    logging.info(f'''Settings:
        Batch size:      {batch_size}
        Validation size: {len(val_loader)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    print('### successfully load the pretrained model ###')

    mask_pred, mask_true, image, f1, precision, recall, accuracy, dice = \
        evaluate_folder(netv, val_loader, device)

    logging.info(f'''Mean error metrics:
        Dice:      {dice}
        f1:        {f1}
        precision: {precision}
        recall:    {recall}
        accuracy:  {accuracy}
    ''')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--loadpath', '-f', type=str,
                        default=current_loc.parent/'checkpoint_epoch15.pth',
                        help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--device', '-d', type=str, default=None, help='device number')
    parser.add_argument('--cropsize', '-cz', type=int, default=None, help='crop size')
    parser.add_argument('--logimg', '-lg', type=str, default=True, help='save the img in weights & biases')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    netv = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)
    netv.to(device=device)
    netv.load_state_dict(
        torch.load(args.loadpath,
                   map_location=device))

    netv.to(device=device)
    try:
        validate(net=netv,
                 batch_size=args.batch_size,
                 device=device,
                 img_scale=args.scale,
                 amp=args.amp,
                 cropsize=args.cropsize,
                 logimage=args.logimg
                )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise

