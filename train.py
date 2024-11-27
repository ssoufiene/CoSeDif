import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.optim import AdamW
from model import Unet, CoSeDif
from model.dataset import GenericNpyDataset
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
now = datetime.now()
logdir = "tf_logs/.../" + now.strftime("%Y%m%d-%H%M%S") + "/"

## Parse CLI arguments ##
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-slr', '--scale_lr', action='store_true', help="Whether to scale lr.")
    parser.add_argument('-rt', '--report_to', type=str, default="tensorboard", choices=["tensorboard"],
                        help="Where to log to. Currently only supports tensorboard")
    parser.add_argument('-ld', '--logging_dir', type=str, default="logs", help="Logging dir.")
    parser.add_argument('-od', '--output_dir', type=str, default="output", help="Output dir.")
    parser.add_argument('-mp', '--mixed_precision', type=str, default="bf16", choices=["no", "fp16", "bf16"],
                        help="Whether to do mixed precision")
    parser.add_argument('-ga', '--gradient_accumulation_steps', type=int, default=1,
                        help="The number of gradient accumulation steps.")
  
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='learning rate')
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.95,
                        help='The beta1 parameter for the Adam optimizer.')
    parser.add_argument('-ab2', '--adam_beta2', type=float, default=0.999,
                        help='The beta2 parameter for the Adam optimizer.')
    parser.add_argument('-aw', '--adam_weight_decay', type=float, default=1e-6,
                        help='Weight decay magnitude for the Adam optimizer.')
    parser.add_argument('-ae', '--adam_epsilon', type=float, default=1e-08,
                        help='Epsilon value for the Adam optimizer.')
    parser.add_argument('-ul', '--use_lion', type=bool, default=False, help='use Lion optimizer')
    parser.add_argument('-ic', '--mask_channels', type=int, default=1, help='input channels for training (default: 1)')
    parser.add_argument('-c', '--input_img_channels', type=int, default=1,
                        help='output channels for training (default: 3)')
    parser.add_argument('-is', '--image_size', type=int, default=128, help='input image size (default: 128)')
    parser.add_argument('-dd', '--data_path', default='./data', help='directory of input image')
    parser.add_argument('-d', '--dim', type=int, default=64, help='dim (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs (default: 10000)')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='batch size to train on (default: 8)')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of timesteps (default: 1000)')
    parser.add_argument('-ds', '--dataset', default='generic', help='Dataset to use')
    parser.add_argument('--save_every', type=int, default=10, help='save_every n epochs (default: 100)')
    return parser.parse_args()

condition_directory = r'/home/b4-13/intern/128_12/edges' # using edges+fault to control the generation
original_directory = r'/home/b4-13/intern/128_12/images'


def load_data(args):
    # Load dataset
    if args.dataset == 'generic':
        transform_list = [transforms.ToTensor()]
        transform_train = transforms.Compose(transform_list)
        dataset = GenericNpyDataset(image_directory=condition_directory, mask_directory=original_directory, transform=None, test_flag=False)

    print(f"The length of the dataset is: {len(dataset)}")

    ## Define PyTorch data generator
    training_generator = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True)
    

    return training_generator

def main():
    
    args = parse_args()
    checkpoint_dir = os.path.join(args.output_dir, 'continue_using_edges2')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging_subdir = os.path.join(args.logging_dir, f'date:_{timestamp}')
    logging_dir = os.path.join(args.output_dir, logging_subdir)
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
  
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    if accelerator.is_main_process:
        writer = SummaryWriter(logging_dir)

    ## DEFINE MODEL ##
    model = Unet(
        dim=args.dim,
        image_size=args.image_size,
        dim_mults=(1, 2, 4, 8),
        mask_channels=args.mask_channels,
        input_img_channels=args.input_img_channels,
        self_condition=False
    )

    ## LOAD DATA ##
    data_loader, val_loader = load_data(args)
   
    ## Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
   
    ## TRAIN MODEL ##
    counter = 0
    model, optimizer, data_loader = accelerator.prepare(
        model, optimizer, data_loader
    )
    diffusion = MedSegDiff(
        model,
        timesteps=args.timesteps
    ).to(accelerator.device)

    if args.load_model_from is not None:
        save_dict = torch.load(args.load_model_from)
        diffusion.model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        accelerator.print(f'Loaded from {args.load_model_from}')

    ## Iterate across training loop
    for epoch in range(args.epochs):
        running_loss = 0.0
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        for (condition, original) in tqdm(data_loader):
            with accelerator.accumulate(model):
                loss = diffusion(original, condition)
                running_loss += loss.item() * img.size(0)
                if accelerator.is_main_process:
                    writer.add_scalar('Loss/train', loss.item(), counter)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            counter += 1
        epoch_loss = running_loss / len(data_loader.dataset)
        print('Training Loss : {:.4f}'.format(epoch_loss))

        ## Saving chechpoints ##
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(checkpoint_dir, f'state_dict_epoch_{epoch}_loss_{epoch_loss}.pt'))

         

               

if __name__ == '__main__':
    main()
