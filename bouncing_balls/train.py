from __future__ import print_function
import argparse
import torch
import models_zoo
from cup_generator.model import Generator
from data_load import Loader
from train_loop import TrainLoop
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='CUP reconstruction')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='Batch size for training')
parser.add_argument('--valid-batch-size', type=int, default=128, metavar='N', help='Batch size for validation')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='Learning rate')
parser.add_argument('--add-noise', action='store_true', default=False, help='Activate Gaussian noise for input')
parser.add_argument('--beta1', type=float, default=0.5, metavar='B1', help='Adam optimizer beta1')
parser.add_argument('--beta2', type=float, default=0.999, metavar='B2', help='Adam optimizer beta2')
parser.add_argument('--max-gnorm', type=float, default=10.0, metavar='G', help='Max gradient norm')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable GPU usage')
parser.add_argument('--data-path', type=str, default=None, metavar='Path', help='Path to pre-saved data')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='Epoch to resume from checkpoint')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for saving checkpoints')
parser.add_argument('--generator-path', type=str, default=None, metavar='Path', help='Path to generator checkpoint')
parser.add_argument('--pretrained-path', type=str, default=None, metavar='Path', help='Path to pretrained model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--save-every', type=int, default=5, metavar='N', help='Save model every N epochs')
parser.add_argument('--n-workers', type=int, default=2, help='Number of workers for data loading')
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for logs')

# Data options
parser.add_argument('--im-size', type=int, default=32, metavar='N', help='Image size (H and W)')
parser.add_argument('--n-frames', type=int, default=25, metavar='N', help='Number of frames per sample')
parser.add_argument('--rep-times', type=int, default=4, metavar='N', help='Frame repetition factor')
parser.add_argument('--train-examples', type=int, default=50000, metavar='N', help='Number of training examples')
parser.add_argument('--val-examples', type=int, default=5000, metavar='N', help='Number of validation examples')
parser.add_argument('--mask-path', type=str, default=None, metavar='Path', help='Path to encoding mask')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# ✅ Dataset Preparation
image_dir = "/content/sheared_images"  # Path to sheared images

train_data_set = Loader(image_dir=image_dir, sample_size=args.train_examples)
valid_data_set = Loader(image_dir=image_dir, sample_size=args.val_examples)

train_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
valid_loader = DataLoader(valid_data_set, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.n_workers)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# ✅ Model Setup
model = models_zoo.model_gen(n_frames=args.n_frames, cuda_mode=args.cuda, input_noise=args.add_noise)
generator = Generator().eval()

# Load Generator Checkpoint if available
if args.generator_path:
    gen_state = torch.load(args.generator_path, map_location=lambda storage, loc: storage)
    generator.load_state_dict(gen_state['model_state'])
else:
    print("⚠️ Warning: No generator checkpoint provided. Training will start from scratch.")

# Load Pretrained Model if available
if args.pretrained_path:
    print(f"\nLoading pretrained model from: {args.pretrained_path}\n")
    ckpt = torch.load(args.pretrained_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['model_state'], strict=False)

# Move to GPU if available
if args.cuda:
    model = model.cuda()
    generator = generator.cuda()
    torch.backends.cudnn.benchmark = True

print(model, '\n')
print(generator, '\n')

# ✅ Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# ✅ TensorBoard Logger
writer = SummaryWriter(log_dir=args.logdir) if args.logdir else None

# ✅ Training Loop
trainer = TrainLoop(model, generator, optimizer, train_loader, valid_loader, 
                    max_gnorm=args.max-gnorm, checkpoint_path=args.checkpoint_path, 
                    checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda, logger=writer)

print(args)
trainer.train(n_epochs=args.epochs, save_every=args.save_every)
