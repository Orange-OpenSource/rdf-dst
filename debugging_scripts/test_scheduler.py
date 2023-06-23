import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from utils.custom_schedulers import LinearWarmupScheduler

# Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Define your training setup
def train(model, optimizer, scheduler, num_epochs):
    lr_values = []  # List to store learning rate values
    
    for epoch in range(num_epochs):
        # Perform forward pass, compute loss, and backpropagation
        # ...

        # Update the learning rate
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        lr_values.append(lr)

        # Print or log the learning rate
        print(f"Epoch {epoch + 1}, Learning Rate: {lr}")

        # Continue with the rest of the training loop
        # ...

    return lr_values

# Set up the training parameters
model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)
warmup_steps = 5
total_steps = 10
scheduler = LinearWarmupScheduler(optimizer, warmup_steps, total_steps)
num_epochs = 10

# Train the model and collect learning rate values
lr_values = train(model, optimizer, scheduler, num_epochs)

# Plot the learning rate schedule
plt.plot(range(num_epochs), lr_values)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
#plt.show()
plt.savefig('learning_rate_schedule.png')

