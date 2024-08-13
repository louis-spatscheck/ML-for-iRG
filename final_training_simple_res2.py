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

def discretize_matrix_to_maintain_mean(matrix):
    # Flatten the matrix to work with it as a 1D array
    flat_matrix = matrix.flatten()
    
    # Calculate the target mean
    target_mean = torch.mean(flat_matrix).item()
    
    # Calculate the number of 1s needed to match the mean
    num_elements = flat_matrix.numel()
    num_ones = int((1 + target_mean)*1.03 / 2 * num_elements)
    num_neg_ones = num_elements - num_ones
    
    # Sort the flattened matrix and get the sorted indices
    sorted_indices = torch.argsort(flat_matrix)
    
    # Create an array of -1s and 1s based on the required numbers
    rounded_output = torch.ones(num_elements)
    rounded_output[:num_neg_ones] = -1
    
    # Rearrange the rounded values to match the original order
    discrete_matrix = torch.empty(num_elements)
    discrete_matrix[sorted_indices] = rounded_output
    
    # Reshape back to the original matrix shape
    return discrete_matrix.reshape(matrix.shape)

def check_cuda():
    if torch.cuda.is_available():
        print(f"CUDA is available. Number of devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")



def calculate_ising_energy(state_batch):
    """
    Berechnet die Energie eines Stapels von 2D-Ising-Modell-Zuständen.

    Args:
        state_batch (torch.Tensor): Ein 4D-Tensor der Form [batch_size, 1, height, width],
                                    der den Zustand des Ising-Modells darstellt.
                                    Jeder Eintrag sollte +1 oder -1 sein.

    Returns:
        list: Eine Liste von Energiewerten für jeden Zustand im Stapel.
    """
    J = 1  # Kopplungskonstante
    energies = []
    
    # Prüfen, ob der Zustand ein 4D-Tensor ist
    if state_batch.ndim != 4 or state_batch.shape[1] != 1:
        raise ValueError("Das Eingangs-Array muss ein 4D-Tensor der Form [batch_size, 1, height, width] sein.")
    
    batch_size, _, Lx, Ly = state_batch.shape
    
    # Berechne die Energie für jeden Zustand im Stapel
    for b in range(batch_size):
        state = state_batch[b, 0]  # Extrahiere das 2D-Bild aus dem Stapel
        energy = 0
        
        for x in range(Lx):
            for y in range(Ly):
                S = state[x, y]
                # Periodische Randbedingungen anwenden (Torussphäre)
                nb = state[(x+1) % Lx, y] + state[x, (y+1) % Ly] + state[(x-1) % Lx, y] + state[x, (y-1) % Ly]
                energy += -J * S * nb

        # Da wir jedes Paar von Nachbarn zweimal zählen, teilen wir die Summe durch 2
        energy /= 2.0
        energies.append(energy.item())
    
    return energies


#torch.backends.cudnn.enabled = False

check_cuda()

betaJ = 0.44

path = '/tikhome/lspatscheck/Documents/bsc/simulation_data/final_training/bigger/no_drop'

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

print(np.mean(np.abs(np.mean(config32 - 1.0,axis=(1,2)))))

config16 = np.array(raw_data16) + 1.0

print(np.mean(np.abs(np.mean(config16 - 1.0,axis=(1,2)))))

np.random.seed(50)
config32 = np.random.permutation(config32)
np.random.seed(50)
config16 = np.random.permutation(config16)


def matrix_to_hashable(matrix):
    """Flattens the matrix and converts it to a hashable type (tuple)."""
    return tuple(matrix.flatten())

def find_duplicates(matrices):
    """Finds duplicates in a list of matrices using hashing for efficiency."""
    seen = {}
    duplicates = []
    for matrix in matrices:
        matrix_tuple = matrix_to_hashable(matrix)
        if matrix_tuple in seen:
            duplicates.append(matrix)
        else:
            seen[matrix_tuple] = True
    return duplicates



duplicates = find_duplicates(config16)

if duplicates:
    print(f"Found {len(duplicates)} duplicate matrices.")
else:
    print("No duplicates found.")

config32_train = config32[:data_size]
config16_train = config16[:data_size]


positive_mean_count = np.sum(np.mean(config16_train, axis=(1, 2)) > 1)

print(f"Anzahl der Matrizen mit positivem Mittelwert: {positive_mean_count/len(config16_train)}")

val_data_size = 200

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
combined_loader = DataLoader(combined_dataset, batch_size=2, shuffle= False)

val_loader = DataLoader(val_combined_dataset, batch_size=1, shuffle = False)



num_batches = len(combined_loader) 

print(num_batches)
print(len(val_loader))

#Dropout angepasst

#Define the TransposedConvolutionCNN
class TransposeCNN(nn.Module):  

    def __init__(self): 
        super(TransposeCNN,self).__init__()

        #set of 128 convolutional layers
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
            #nn.Dropout2d(0.5),
            nn.ReLU(inplace = True)  
        )


        

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
            nn.ReLU(inplace = True)  
        ) 

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
 
        ) 

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
            nn.ReLU(inplace = True)  
        ) 

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
        ) 

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
            nn.ReLU(inplace = True) 
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
            nn.ReLU(inplace = True) 
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
            nn.ReLU(inplace = True) 
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht 
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
            nn.ReLU(inplace = True) 
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht

        )

        self.conv13 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht
            nn.ReLU(inplace = True) 
        )

        self.conv14 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(256),  # Batch-Normalisierungsschicht

        )

        self.conv15 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,padding_mode="circular"),
            nn.BatchNorm2d(128),  # Batch-Normalisierungsschicht
            nn.ReLU(inplace = True) 
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, stride=1,padding=1,padding_mode="circular")
        )


    def forward(self,x):

        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)

        x = nn.functional.relu(x + shortcut)

        x = self.layer1(x)


        shortcut = x
        x = self.conv3(x)
        x = self.conv4(x)

        x = nn.functional.relu(x + shortcut)

        shortcut = x
        x = self.conv5(x)
        x = self.conv6(x)

        x = nn.functional.relu(x + shortcut)

        x = self.final_conv(x)

        return x
    

repeat = 30
total_repeats =36

while repeat < total_repeats:

    model = TransposeCNN()

    #define the loss function
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    #optimizer = optim.SGD(model.parameters(), lr=0.00005,momentum=0.9)
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
        energy_losses = []
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
            train_energy_loss = 0.0
            for originals, inputs in combined_loader:
                #originals = originals.view(-1, 1, 32, 32)
                #inputs = inputs.view(-1, 1, 16, 16)

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
                    
                    if epoch % 5 ==0:




                        mean_originals = torch.mean(val_originals-1.0)
                        mean_outputs = torch.mean(val_outputs-1.0)


                        #val_disoutputs = discretize_matrix_to_maintain_mean(val_outputs-1.0)

                        #energy_inputs = np.mean(calculate_ising_energy(val_originals - 1.0) )/16**2
                        #energy_outputs = np.mean(calculate_ising_energy(val_disoutputs))/32**2


                        mag_loss += float(torch.abs(mean_originals) - torch.abs(mean_outputs))/float(torch.abs(mean_originals))
                        #energy_loss += (energy_inputs- energy_outputs) /energy_inputs
                        
                        

                        





            if epoch % 5 == 0:
                mag_losses.append( mag_loss / (len(val_loader) ) )
                energy_losses.append( energy_loss / (len(val_loader) ) )
            val_losses[epoch] =  val_loss / (len(val_loader) )


            epoch_loss = running_loss / num_batches

            losses[epoch] = epoch_loss  # store the loss value
            if epoch % 2 ==1:
                #print('Epoch [%d] loss: %.6f' % (epoch + 1, epoch_loss))
                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {losses[epoch]}, Validation Loss: {val_losses[epoch]}, Mean mag: {mag_losses[-1]}, Energy loss: {energy_losses[-1]}")
            







        # Plot the loss function
        plt.figure()
        plt.plot(losses,label = "loss")
        plt.plot(val_losses, label = "val_loss")
        #plt.plot(np.arange(0,epochs,10),np.abs(mag_losses), label = "mag_loss")
        plt.plot(np.arange(0,epochs,5),np.abs(energy_losses), label = "energy_loss")
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

        pickle.dump(
            energy_losses,
            open(
                f'{path}/run_{repeat}/losses_data/loss_energy_{training}.pickle',
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

                # Bestimme das Vorzeichen jedes Elements
                #signs = torch.sign(outputs)

        # Setze die positiven Werte auf 1 und die negativen Werte auf -1
                #outputs = torch.where(signs > 0, torch.tensor(1.0), torch.tensor(-1.0))


                origin = originals.cpu().squeeze().numpy()
                out = outputs.cpu().squeeze().numpy()
                inp = inputs.cpu().squeeze().numpy()

                rounded_output = np.where(out >= 1, 2.0, 0.0)


                # Finde den maximalen und minimalen Wert in beiden Bildern
                min_val = min(origin.min(), out.min())
                max_val = max(origin.max(), out.max())

                # Erstelle einen neuen Plot
                plt.figure(figsize=(20, 8))

                # Plotte das Zielbild auf der linken Seite
                plt.subplot(1, 5, 1)
                plt.imshow(origin, cmap='gray', vmin=min_val, vmax=max_val)
                plt.title('Original')

                # Plotte das Vorhersagebild auf der rechten Seite
                plt.subplot(1, 5, 2)
                plt.imshow(out, cmap='gray', vmin=min_val, vmax=max_val)
                plt.title('Output')

                plt.subplot(1, 5, 3)
                plt.imshow(inp, cmap='gray', vmin=min_val, vmax=max_val)
                plt.title('Input')


                errors = (out - origin)**2 
                rounded_errors = (rounded_output - origin)**2 

                plt.subplot(1, 5, 4)
                #plt.imshow(errors.mean(dim=0).squeeze(), cmap='seismic', interpolation='nearest')
                plt.imshow(rounded_errors, cmap='seismic', interpolation='nearest')
                plt.colorbar(label='Rounded Error')
                plt.title(f'Rounded Error: {np.mean(rounded_errors)}')

                plt.subplot(1, 5, 5)
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

                # Bestimme das Vorzeichen jedes Elements
                #signs = torch.sign(outputs)

        # Setze die positiven Werte auf 1 und die negativen Werte auf -1
                #outputs = torch.where(signs > 0, torch.tensor(1.0), torch.tensor(-1.0))


                origin = originals.cpu().squeeze().numpy()
                out = outputs.cpu().squeeze().numpy()
                inp = inputs.cpu().squeeze().numpy()

                rounded_output = np.where(out >= 1, 2.0, 0.0)


                # Finde den maximalen und minimalen Wert in beiden Bildern
                min_val = min(origin.min(), out.min())
                max_val = max(origin.max(), out.max())

                # Erstelle einen neuen Plot
                plt.figure(figsize=(20, 8))

                # Plotte das Zielbild auf der linken Seite
                plt.subplot(1, 5, 1)
                plt.imshow(origin, cmap='gray', vmin=min_val, vmax=max_val)
                plt.title('Original')

                # Plotte das Vorhersagebild auf der rechten Seite
                plt.subplot(1, 5, 2)
                plt.imshow(out, cmap='gray', vmin=min_val, vmax=max_val)
                plt.title('Output')

                plt.subplot(1, 5, 3)
                plt.imshow(inp, cmap='gray', vmin=min_val, vmax=max_val)
                plt.title('Input')


                errors = (out - origin)**2 
                rounded_errors = (rounded_output - origin)**2 

                plt.subplot(1, 5, 4)
                #plt.imshow(errors.mean(dim=0).squeeze(), cmap='seismic', interpolation='nearest')
                plt.imshow(rounded_errors, cmap='seismic', interpolation='nearest')
                plt.colorbar(label='Rounded Error')
                plt.title(f'Rounded Error: {np.mean(rounded_errors)}')

                plt.subplot(1, 5, 5)
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