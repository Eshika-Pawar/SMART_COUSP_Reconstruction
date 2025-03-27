from __future__ import print_function
import argparse
import torch
import models_zoo
from cup_generator.model import Generator
from data_load import Loader, Loader_offline
from train_loop import TrainLoop
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='CUP reconstruction')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=128, help='Batch size for validation (default: 128)')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate (default: 0.0003)')
parser.add_argument('--add-noise', action='store_true', default=False, help='Activates Gaussian noise in inputs')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta 1 (default: 0.5)')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta 2 (default: 0.999)')
parser.add_argument('--max-gnorm', type=float, default=10.0, help='Max gradient norm (default: 10.0)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable GPU use')
parser.add_argument('--data-path', type=str, default=None, help='Path to pre-saved data')
parser.add_argument('--checkpoint-epoch', type=int, default=None, help='Epoch to load for checkpointing')
parser.add_argument('--checkpoint-path', type=str, default=None, help='Path for checkpointing')
parser.add_argument('--generator-path', type=str, default=None, help='Path for generator params')
parser.add_argument('--pretrained-path', type=str, default=None, help='Path to trained model')
parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=5, help='Batches before logging training status')
parser.add_argument('--n-workers', type=int, default=4, help='Number of workers for DataLoader (default: 4)')
parser.add_argument('--logdir', type=str, default=None, help='Path for logging')
parser.add_argument('--im-size', type=int, default=32, help='Frame size (default: 32)')
parser.add_argument('--n-frames', type=int, default=25, help='Number of frames per sample (default: 25)')
parser.add_argument('--rep-times', type=int, default=4, help='Frame repetition (default: 4)')
parser.add_argument('--train-examples', type=int, default=50000, help='Number of training examples (default: 50000)')
parser.add_argument('--val-examples', type=int, default=5000, help='Number of validation examples (default: 5000)')
parser.add_argument('--mask-path', type=str, default=None, help='Path to encoding mask')

args = parser.parse_args()

# Enable CUDA if available
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    print("✅ CUDA is enabled!")
else:
    print("⚠️ Running on CPU! Training might be slow.")

# Define dataset paths
image_dir = "/content/sheared_images"  # Path to sheared images
video_dir = "/content/droplet_burst.mat"  # Path to clear video

# Load dataset
print("✅ Loading dataset...")
train_data_set = Loader(image_dir=image_dir, sample_size=args.train_examples)
valid_data_set = Loader(image_dir=image_dir, sample_size=args.val_examples)

# Create DataLoaders
train_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
valid_loader = DataLoader(valid_data_set, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.n_workers)

# Debug: Check if DataLoader works
print("✅ Checking training DataLoader...")
for batch_idx, (sheared_image, clear_video) in enumerate(train_loader):
    print(f"Batch {batch_idx}: Sheared Image Shape: {sheared_image.shape}, Clear Video Shape: {clear_video.shape}")
    if batch_idx == 5:  # Stop early after 5 batches
        break
print("✅ DataLoader is working correctly!")

# Set random seed for reproducibility
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load models
print("✅ Initializing models...")
model = models_zoo.model_gen(n_frames=10, cuda_mode=args.cuda, input_noise=args.add_noise)
generator = Generator().eval()

# Load generator checkpoint if provided
if args.generator_path:
    print(f"✅ Loading generator checkpoint from {args.generator_path}")
    gen_state = torch.load(args.generator_path, map_location=torch.device('cuda' if args.cuda else 'cpu'))
    generator.load_state_dict(gen_state['model_state'])
else:
    print("⚠️ No generator checkpoint provided. Training will start from scratch.")

# Load pretrained model
if args.pretrained_path:
    print(f"✅ Loading pretrained model from {args.pretrained_path}")
    ckpt = torch.load(args.pretrained_path, map_location=torch.device('cuda' if args.cuda else 'cpu'))
    model.load_state_dict(ckpt['model_state'], strict=False)

# Move models to GPU if CUDA is enabled
if args.cuda:
    model = model.cuda()
    generator = generator.cuda()
    torch.backends.cudnn.benchmark = True

# Print model summaries
print(model)
print(generator)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# Setup TensorBoard logging
writer = SummaryWriter(log_dir=args.logdir) if args.logdir else None

# Initialize training loop
trainer = TrainLoop(
    model, generator, optimizer, train_loader, valid_loader,
    max_gnorm=args.max_gnorm, checkpoint_path=args.checkpoint_path,
    checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda, logger=writer
)

# Start training
print(f"🚀 Starting training for {args.epochs} epochs...")
trainer.train(n_epochs=args.epochs, save_every=args.save_every)
