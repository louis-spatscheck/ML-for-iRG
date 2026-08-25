import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import time
from torch.utils.data import DataLoader, TensorDataset
import argparse
import torch

#the paths to load data and models, aswell as save models and losses have to be added

path = '/tikhome/lspatscheck/Documents/bsc/simulation_data/final_training/bigger/no_drop'  #path to the training data

parser = argparse.ArgumentParser(description="Model Training")
parser.add_argument("--sample_size", type=int, help="Sample Size")


args = parser.parse_args()
data_size = args.sample_size


raw_data32= pickle.load(
open(
     f'{path}/train_data/config.pickle',
     'rb'
    )
)


raw_data16= pickle.load(
open(
     f'{path}/train_data/config_renorm.pickle',
     'rb'
    )
)
config32 = np.array(raw_data32['L=32 configurations'])+ 1.0
config16 = np.array(raw_data16) + 1.0                  # load the configurations and add 1.0 to ensure that the ReLu works correctly

np.random.seed(50)
config32 = np.random.permutation(config32)
np.random.seed(50)
config16 = np.random.permutation(config16)


config32_train = config32[:data_size]
config16_train = config16[:data_size]


val_data_size = int(data_size *0.25)

config32_val = config32[-val_data_size:]
config16_val = config16[-val_data_size:]





val_combined_dataset = TensorDataset(
    torch.tensor(config32_val, dtype=torch.float32).unsqueeze(1),  # original val_data
    torch.tensor(config16_val, dtype=torch.float32).unsqueeze(1)   # input val_data
)


# Combine the original and input data into a single dataset
combined_dataset = TensorDataset(
    torch.tensor(config32_train, dtype=torch.float32).unsqueeze(1),  # original train_data
    torch.tensor(config16_train, dtype=torch.float32).unsqueeze(1)   # input train_data
)

# Create a DataLoader for the combined dataset
combined_loader = DataLoader(combined_dataset, batch_size=1, shuffle= False)

val_loader = DataLoader(val_combined_dataset, batch_size=1, shuffle = False)

num_batches = len(combined_loader) 

#Define the TransposedConvolutionCNN
class TransposeCNN(nn.Module):  

    def __init__(self): 
        super(TransposeCNN,self).__init__()

        #set of 128 convolutional layers
        self.Tconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
            #nn.Dropout2d(0.2),
            nn.ReLU(inplace = True)  
        )

        self.Tconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),  # Batch-Normalisierungsschicht
            #nn.Dropout2d(0.2),
            nn.ReLU(inplace = True)  
        )


        self.Tconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),  # Batch-Normalisierungsschicht
            #nn.Dropout2d(0.2),
            nn.ReLU(inplace = True)  
        )

        self.pooling = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace = True) 
        )

        self.pooling2 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace = True) 
        )

        

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(128),  # Batch-Normalisierungsschicht
            nn.ReLU(inplace = True)  
        ) 

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
            nn.ReLU(inplace = True)  
        ) 

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(512),  # Batch-Normalisierungsschicht
            #nn.Dropout2d(0.5),
            nn.ReLU(inplace = True)  
        ) 

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(512),  # Batch-Normalisierungsschicht
            nn.ReLU(inplace = True)  #added ReLu
        ) 

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(512),  # Batch-Normalisierungsschicht
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
            #nn.Dropout2d(0.5),
            nn.ReLU(inplace = True)  #added ReLu
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(128),  # Batch-Normalisierungsschicht
            #nn.Dropout2d(0.5),
            nn.ReLU(inplace = True)  
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(64),  # Batch-Normalisierungsschicht
            #nn.Dropout2d(0.5),
            nn.ReLU(inplace = True) 
        )


        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1,padding=0,padding_mode="circular")
        )


    def forward(self,x):

        x = self.conv1(x)
        
        shortcut1 = x

        x = self.pooling(x)


        x = self.conv2(x)

        shortcut2 = x

        x = self.pooling2(x)


        x = self.conv3(x)

        shortcut3 = x


        x = self.conv4(x)
        x = self.conv5(x)

        x = nn.functional.relu( x + shortcut3)



        x = self.Tconv1(x)

        x = torch.cat((x, shortcut2), dim=1)

        x = self.conv6(x)


        x = self.Tconv2(x)

        x = torch.cat((x, shortcut1), dim=1)

        x = self.conv7(x)

        x = self.Tconv3(x)

        x = self.conv8(x)

        x = self.final_conv(x)

        return x




repeat = 0
total_repeats = 10

while repeat < total_repeats:

    model = TransposeCNN()

    #define the loss function
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    print("Start Training")




    train_time = 10
    training = 0

    while training < train_time:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if training > 0:

            model.load_state_dict(torch.load(f"{path}/run_{repeat}/models/model_{training-1}.pth",map_location=device))
            model.to(device)




        epochs = 100
        losses = np.empty(epochs,dtype=np.float32)  # list to store the loss values
        val_losses = np.empty(epochs,dtype=np.float32)
        mag_losses = []
        batch_count = 0
        k = 0

        for epoch in range(epochs):      # loop over dataset multiple times
            running_loss = 0.0
            model.train()
            if epoch == 0:
                start = time.time()

            if epoch == 6:
                end_time = (time.time()- start)/6.0
                print("Estimated Time [h]:",end_time*epochs /3600)

            # Loop over your data in batches
            for originals, inputs in combined_loader:

                inputs, originals = inputs.to(device), originals.to(device)

                # zero the parameter gradients

                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)



                loss = criterion(outputs, originals)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()


            model.eval()
            n = 0
            with torch.no_grad():
                val_loss = 0.0
                mag_loss = 0.0
                energy_loss = 0.0
                for val_originals,val_inputs in val_loader:

                    val_inputs, val_originals = val_inputs.to(device), val_originals.to(device)
                    val_outputs = model(val_inputs)


                    loss = criterion(val_outputs, val_originals)
                    val_loss += loss.item()
                    
                    if epoch % 10 ==0: #additional monitoring of the magnetization during the trainig


                                                

                        mean_originals = torch.mean(val_originals-1.0)
                        mean_outputs = torch.mean(val_outputs-1.0)



                        mag_loss += float(torch.abs(mean_originals) - torch.abs(mean_outputs))
     
                        
                        

                      




            if epoch % 10 == 0:
                mag_losses.append( mag_loss / (len(val_loader) ) )
            val_losses[epoch] =  val_loss / (len(val_loader) )


            epoch_loss = running_loss / num_batches

            losses[epoch] = epoch_loss  # store the loss value
            if epoch % 2 ==1:
                #print('Epoch [%d] loss: %.6f' % (epoch + 1, epoch_loss))
                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {losses[epoch]}, Validation Loss: {val_losses[epoch]}, Mean mag: {mag_losses[-1]}")




        # Plot the loss function
        plt.figure()
        plt.plot(losses,label = "loss")
        plt.plot(val_losses, label = "val_loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Function')
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.legend()
        plt.savefig(f"{path}/run_{repeat}/losses_plot/training_loss_{training}.png")








        pickle.dump(
            losses,
            open(
                f"{path}/run_{repeat}/losses_data/losses_{training}.pickle",
                mode = 'wb'
            )
        )

        pickle.dump(
            val_losses,
            open(
                f'{path}/run_{repeat}/losses_data/val_losses_{training}.pickle',
                mode = 'wb'
            )
        )

        pickle.dump(
            mag_losses,
            open(
                f'{path}/run_{repeat}/losses_data/loss_mag_{training}.pickle',
                mode = 'wb'
            )
        )



        print('Finished Training')


        # Annahme: Sie haben ein trainiertes Modell namens 'model'
        torch.save(model.state_dict(), f"{path}/run_{repeat}/models/model_{training}.pth")



        correct = 0
        total = 0
        with torch.no_grad():
            for i in [0,val_data_size//2 -1,val_data_size - 1]:



                original_tensor = torch.tensor(config32_val[i], dtype=torch.float32)
                originals = torch.reshape(original_tensor,(1,1,32,32))


                input_tensor = torch.tensor(config16_val[i], dtype=torch.float32)
                inputs = torch.reshape(input_tensor,(1,1,16,16))

                inputs, originals  = inputs.to(device) , originals.to(device)

                outputs = model(inputs)



                origin = originals.cpu().squeeze().numpy()
                out = outputs.cpu().squeeze().numpy()
                inp = inputs.cpu().squeeze().numpy()



                # Finde den maximalen und minimalen Wert in beiden Bildern
                min_val = min(origin.min(), out.min())
                max_val = max(origin.max(), out.max())

                # Erstelle einen neuen Plot
                plt.figure(figsize=(20, 8))

                # Plotte das Zielbild auf der linken Seite
                plt.subplot(1, 4, 1)
                plt.imshow(origin, cmap='gray', vmin=min_val, vmax=max_val)
                plt.title('Original')

                # Plotte das Vorhersagebild auf der rechten Seite
                plt.subplot(1, 4, 2)
                plt.imshow(out, cmap='gray', vmin=min_val, vmax=max_val)
                plt.title('Output')

                plt.subplot(1, 4, 3)
                plt.imshow(inp, cmap='gray', vmin=min_val, vmax=max_val)
                plt.title('Input')


                errors = (out - origin)**2 


                plt.subplot(1, 4, 4)
                #plt.imshow(errors.mean(dim=0).squeeze(), cmap='seismic', interpolation='nearest')
                plt.imshow(errors, cmap='seismic', interpolation='nearest')
                plt.colorbar(label='Error')
                plt.title('Error for Each Pixel')


                # Zeige den Plot an

                plt.savefig(f'{path}/run_{repeat}/pictures/pictures_val_{training}_{i}.png')

        with torch.no_grad():
            for i in [0,data_size//2 -1,data_size - 1]:

                original_tensor = torch.tensor(config32_train[i], dtype=torch.float32)
                originals = torch.reshape(original_tensor,(1,1,32,32))


                input_tensor = torch.tensor(config16_train[i], dtype=torch.float32)
                inputs = torch.reshape(input_tensor,(1,1,16,16))

                inputs, originals  = inputs.to(device) , originals.to(device)

                outputs = model(inputs)


                origin = originals.cpu().squeeze().numpy()
                out = outputs.cpu().squeeze().numpy()
                inp = inputs.cpu().squeeze().numpy()


                # Finde den maximalen und minimalen Wert in beiden Bildern
                min_val = min(origin.min(), out.min())
                max_val = max(origin.max(), out.max())

                # Erstelle einen neuen Plot
                plt.figure(figsize=(20, 8))

                # Plotte das Zielbild auf der linken Seite
                plt.subplot(1, 4, 1)
                plt.imshow(origin, cmap='gray', vmin=min_val, vmax=max_val)
                plt.title('Original')

                # Plotte das Vorhersagebild auf der rechten Seite
                plt.subplot(1, 4, 2)
                plt.imshow(out, cmap='gray', vmin=min_val, vmax=max_val)
                plt.title('Output')

                plt.subplot(1, 4, 3)
                plt.imshow(inp, cmap='gray', vmin=min_val, vmax=max_val)
                plt.title('Input')


                errors = (out - origin)**2 


                plt.subplot(1, 4, 4)
                #plt.imshow(errors.mean(dim=0).squeeze(), cmap='seismic', interpolation='nearest')
                plt.imshow(errors, cmap='seismic', interpolation='nearest')
                plt.colorbar(label='Error')
                plt.title('Error for Each Pixel')


                # Zeige den Plot an

                plt.savefig(f'{path}/run_{repeat}/pictures/pictures_train_{training}_{i}.png')

                plt.close('all')



        training += 1

    print("Finished Training")
    repeat += 1
            

