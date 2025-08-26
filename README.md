# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: HARISHKUMAR R
### Register Number: 212223230073
```python

class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu=nn.ReLU()
        self.history={'loss:':[]}

  def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

harish=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(harish.parameters(),lr=0.001)


def train_model(harish, X_train, y_train, criterion, optimizer, epochs=1000):
    # initialize history before loop
    harish.history = {'loss': []}

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = harish(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # record loss
        harish.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
train_model(harish, X_train_tensor, y_train_tensor, criterion,optimizer)

with torch.no_grad():
    test_loss = criterion(harish(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')


import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame from the loss history
loss_df = pd.DataFrame(harish.history['loss'], columns=['Loss'])

# Plot the loss curve
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()



X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = harish(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```
## Dataset Information
![alt text](<Screenshot 2025-08-26 091946.png>)

## OUTPUT

### Training Loss Vs Iteration Plot
![alt text](<Screenshot 2025-08-26 091931.png>)

### New Sample Data Prediction
![alt text](<Screenshot 2025-08-26 091938.png>)
## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.