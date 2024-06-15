import torch
import torch.nn as nn
import argparse
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from matplotlib import pyplot as plt

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # 128->256->512->784
        self.model = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=784),
            nn.BatchNorm1d(784),
            nn.Tanh(),
        )
        
    def forward(self, x):
        # 784->28*28
        generated_image = self.model(x).view(-1, 1, 28, 28)
        return generated_image
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 784->512->256->128->1
        self.model = nn.Sequential(
            nn.Linear(in_features=784,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 28*28->784
        flatten_x = x.view(-1, 784)
        score = self.model(flatten_x)
        return score


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True, help='Enables CUDA (GPU) if available')
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="adam: learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam: decay rate of first-order momentum of gradient")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam: decay rate of second-order momentum of gradient")
    args = parser.parse_args()
    args_dict = vars(args)  
    hyper_parameters_str = ", ".join([f"{key}: {value}" for key, value in args_dict.items()])
    
    # cuda
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # load data
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    
    data = datasets.MNIST(
        "./data", 
        train=True, 
        transform=transformer, 
        download=True
    )
    dataloader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    # get model
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator_optimizer = torch.optim.Adam(params=generator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2)) 
    discriminator_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2)) 
    
    # loss function
    loss_fn = torch.nn.BCELoss()
    
    #
    gen = torch.Generator(device).manual_seed(42)
    tqdm.write("Training Start!")
    tqdm.write("Hyper Parameters: " + hyper_parameters_str)
    global_generator_loss = []
    global_discriminator_loss = []
    global_real_score = []
    global_fake_score = []
    for epoch in range(args.epochs):
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        loop.set_description(f'Epoch [{epoch+1}/{args.epochs}]')
        for steps, (real_image, label) in loop:
            # Train Dis
            if steps % 20 == 0:
                # real case
                real_image = real_image.to(device)
                real_score = discriminator(real_image)
                real_loss = loss_fn(real_score, torch.ones_like(real_score))
                
                # fake case
                fake_vector = torch.randn(size=(args.batch_size, 128), device=device, generator=gen)
                fake_image = generator(fake_vector)
                fake_score = discriminator(fake_image)
                fake_loss = loss_fn(fake_score, torch.zeros_like(fake_score))
                
                # sum, backprop
                discriminator_loss = real_loss + fake_loss
                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                discriminator_optimizer.step()
                with torch.no_grad():
                    global_discriminator_loss.append(discriminator_loss.item())
                    global_real_score.append(real_score.data.mean())
                
            # Train Gen
            fake_vector = torch.randn(size=(args.batch_size, 128), device=device, generator=gen)
            fake_image = generator(fake_vector)
            fake_score = discriminator(fake_image)
            generator_loss = loss_fn(fake_score, torch.ones_like(fake_score))
            # backprop
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()
            
            with torch.no_grad():
                global_generator_loss.append(generator_loss.item())
                global_fake_score.append(fake_score.data.mean())
            
            loop.set_postfix(gen_loss = generator_loss.item(), dis_loss = discriminator_loss.item())
            
            if steps == len(dataloader)-1:
                save_image(fake_image.data[:25], "images/epoch_%d.png" % (epoch+1), nrow=5, normalize=True)
            
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                avg_generator_loss = sum(global_generator_loss)/len(global_generator_loss)
                avg_discriminator_loss = sum(global_discriminator_loss)/len(global_discriminator_loss)
                avg_real_score = sum(global_real_score)/len(global_real_score)
                avg_fake_score = sum(global_fake_score)/len(global_fake_score)
                global_generator_loss = []
                global_discriminator_loss = []
                global_real_score = []
                global_fake_score = []
            loop.write(
                "[Epoch %d-%d, avg_generator_loss: %f, avg_discriminator_loss: %f, avg_real_score: %f, avg_fake_score: %f]"
                % (epoch-8, epoch+1, avg_generator_loss, avg_discriminator_loss, avg_real_score, avg_fake_score)
            )
    
    tqdm.write("Training End!")
    torch.save(generator, './models/generator.pth')
    torch.save(discriminator, './models/discriminator.pth')

    
if __name__ == "__main__":
    
    # train()
    generator = torch.load("./models/generator.pth", map_location="cpu")
    generator.eval()
    img = generator(torch.randn(size=(1, 128))).detach().numpy().squeeze()
    plt.imshow(img, cmap='gray')  
    plt.axis('off')  
    plt.show()
