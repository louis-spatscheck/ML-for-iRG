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


import autocorr  # from this repository



from tqdm import tqdm




betaJ = 0.44

model = "UNet_ResNet"
sample_size = 10000
small = 16
big = 2 * small
runs = ["run_1","run_2"] 

alpha = np.ones(len(runs))  ### the correct values of alpha have to be chosen according to the results
for l,run in enumerate(runs):





    def discretize_matrix_to_maintain_mean(matrix):
        # Flatten the matrix to work with it as a 1D tensor
        flat_matrix = matrix.flatten()
        
        # Calculate the target mean
        target_mean = torch.mean(flat_matrix)
        
        # Calculate the number of 1s needed to match the mean
        num_elements = flat_matrix.numel()
        num_ones = int((1 + (target_mean.item()*alpha[l])) / 2 * num_elements)
        num_neg_ones = num_elements - num_ones
        
        # Sort the flattened matrix and get the sorted indices
        sorted_indices = torch.argsort(flat_matrix)
        
        # Create an array of -1s and 1s based on the required numbers
        rounded_output = torch.ones(num_elements, device=matrix.device)
        rounded_output[:num_neg_ones] = -1
        
        # Rearrange the rounded values to match the original order
        discrete_matrix = torch.empty(num_elements, device=matrix.device)
        discrete_matrix[sorted_indices] = rounded_output
        
        # Reshape back to the original matrix shape
        return discrete_matrix.reshape(matrix.shape)

    import os

############# the paths to the models and data of L=16 and L=32 simulations have to be define by yourself

    model_directory = f"/tikhome/lspatscheck/Documents/bsc/simulation_data/final_models/complex_UNet_20000/{run}/models/model_17.pth"
    path = f"/data/lspatscheck/complexUNet2000/{run}/test" 

    if os.path.isdir(path):
        print(f"The directory {path} exists.")
    else:
        print(f"The directory {path} does not exist.")




    raw_data_small= pickle.load(
    open(
            f"/data/lspatscheck/test_samples/test_data{small}.pickle",
        'rb'
        )
    )

    print("Done")
    




    config_small = np.array(raw_data_small[f'L={small} configurations'])





    np.random.seed(55)
    config_small = np.random.permutation(config_small)[:sample_size] + 1.0



    print(np.mean(np.abs(np.mean(config_small - 1.0,axis=(1,2)))))
    


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
        

    for i in range(0,10):
        nested_folder = f"{path}/{small}_{big*2**i}"

        if not os.path.exists(nested_folder):
            os.makedirs(nested_folder)
            print(f"Verschachtelter Ordner '{nested_folder}' wurde erstellt.")

    model = TransposeCNN()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in [5000]:
        data_size = i
        print("Hi")
        model.load_state_dict(torch.load(model_directory))

        model.to(device)

        model.eval()


        print('Finished Training')



        config_small_test = config_small

        correct = 0
        total = 0

        mag1 = np.empty(sample_size,dtype= np.float32)
        mag2 = np.empty(sample_size,dtype= np.float32)
        mag3 = np.empty(sample_size,dtype= np.float32)
        mag4 = np.empty(sample_size,dtype= np.float32)
        mag5 = np.empty(sample_size,dtype= np.float32)
        mag6 = np.empty(sample_size,dtype= np.float32)
        mag7 = np.empty(sample_size,dtype= np.float32)
        #mag8 = np.empty(sample_size,dtype= np.float32)

        mag1_round = np.empty(sample_size,dtype= np.float32)
        mag2_round = np.empty(sample_size,dtype= np.float32)
        mag3_round = np.empty(sample_size,dtype= np.float32)
        mag4_round = np.empty(sample_size,dtype= np.float32)
        mag5_round = np.empty(sample_size,dtype= np.float32)
        mag6_round = np.empty(sample_size,dtype= np.float32)
        mag7_round = np.empty(sample_size,dtype= np.float32)
        #mag8_round = np.empty(sample_size,dtype= np.float32)

        with torch.no_grad():
            for k in tqdm(range(len(config_small_test)), desc="Verarbeitung"):



                input_tensor = torch.tensor(config_small_test[k], dtype=torch.float32)
                inputs = torch.reshape(input_tensor,(1,1,small,small))

                inputs= inputs.to(device)

                outputs = model(inputs)
                rounded_output = discretize_matrix_to_maintain_mean(outputs-1.0)


                outputs2 = model(rounded_output+1.0)
                rounded_output2 = discretize_matrix_to_maintain_mean(outputs2-1.0)

                outputs3 = model(rounded_output2+1.0)
                rounded_output3 = discretize_matrix_to_maintain_mean(outputs3-1.0)


                outputs4 = model(rounded_output3+1.0)
                rounded_output4 = discretize_matrix_to_maintain_mean(outputs4-1.0)

                outputs5 = model(rounded_output4+1.0)
                rounded_output5 = discretize_matrix_to_maintain_mean(outputs5-1.0)

                outputs6 = model(rounded_output5+1.0)
                rounded_output6 = discretize_matrix_to_maintain_mean(outputs6-1.0)
                
                outputs7 = model(rounded_output6+1.0)
                rounded_output7 = discretize_matrix_to_maintain_mean(outputs7 -1.0)

                #outputs8 = model(rounded_output7+1.0)
                #rounded_output8 = discretize_matrix_to_maintain_mean(outputs8-1.0)
                


                # Bestimme das Vorzeichen jedes Elements
                #signs = torch.sign(outputs)

        # Setze die positiven Werte auf 1 und die negativen Werte auf -1
                #outputs = torch.where(signs > 0, torch.tensor(1.0), torch.tensor(-1.0))


                #origin = originals.cpu().squeeze().numpy() 
                inp = inputs.cpu().squeeze().numpy() -1.0

                out1 = outputs.cpu().squeeze().numpy() -1.0
                out2 = outputs2.cpu().squeeze().numpy() -1.0
                out3 = outputs3.cpu().squeeze().numpy() -1.0
                out4 = outputs4.cpu().squeeze().numpy() -1.0
                out5 = outputs5.cpu().squeeze().numpy() -1.0
                out6 = outputs6.cpu().squeeze().numpy() -1.0
                out7 = outputs7.cpu().squeeze().numpy() -1.0
                #out8 = outputs8.cpu().squeeze().numpy() -1.0

                round_out1 = rounded_output.cpu().squeeze().numpy() 
                round_out2 = rounded_output2.cpu().squeeze().numpy() 
                round_out3 = rounded_output3.cpu().squeeze().numpy() 
                round_out4 = rounded_output4.cpu().squeeze().numpy() 
                round_out5 = rounded_output5.cpu().squeeze().numpy() 
                round_out6 = rounded_output6.cpu().squeeze().numpy() 
                round_out7 = rounded_output7.cpu().squeeze().numpy() 
                #round_out8 = rounded_output8.cpu().squeeze().numpy() 


                mag1[k] = np.mean(out1)
                mag2[k] = np.mean(out2)
                mag3[k] = np.mean(out3)
                mag4[k]= np.mean(out4)
                mag5[k] = np.mean(out5)
                mag6[k] = np.mean(out6)
                mag7[k] = np.mean(out7)
                #mag8[k]= np.mean(out8)

                mag1_round[k] = np.mean(round_out1)
                mag2_round[k] = np.mean(round_out2)
                mag3_round[k] = np.mean(round_out3)
                mag4_round[k] = np.mean(round_out4)
                mag5_round[k] = np.mean(round_out5)
                mag6_round[k] = np.mean(round_out6)
                mag7_round[k] = np.mean(round_out7)
                #mag8_round[k] = np.mean(round_out8)


                if k % 2000 == 0:


                    min_val = -1.0
                    max_val = 1.0
                    
                    #Erstelle einen neuen Plot
                    plt.figure(figsize=(80,30))

                    plt.subplot(1, 17, 1)
                    plt.imshow(inp, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Input')

                    # Plotte das Vorhersagebild auf der rechten Seite
                    plt.subplot(1, 17, 2)
                    plt.imshow(out1, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Continues Output')

                    # Plotte das Vorhersagebild auf der rechten Seite
                    plt.subplot(1, 17, 3)
                    plt.imshow(round_out1, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Discrete Output')

                    plt.subplot(1, 17, 4)
                    plt.imshow(out2, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Continues Output')

                    # Plotte das Vorhersagebild auf der rechten Seite
                    plt.subplot(1, 17, 5)
                    plt.imshow(round_out2, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Discrete Output')


                    plt.subplot(1, 17, 6)
                    plt.imshow(out3, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Continues Output')

                    # Plotte das Vorhersagebild auf der rechten Seite
                    plt.subplot(1, 17, 7)
                    plt.imshow(round_out3, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Discrete Output')


                    plt.subplot(1, 17, 8)
                    plt.imshow(out4, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Continues Output')

                    # Plotte das Vorhersagebild auf der rechten Seite
                    plt.subplot(1, 17, 9)
                    plt.imshow(round_out4, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Discrete Output')

                    plt.subplot(1, 17, 10)
                    plt.imshow(out5, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Continues Output')

                    # Plotte das Vorhersagebild auf der rechten Seite
                    plt.subplot(1, 17, 11)
                    plt.imshow(round_out5, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Discrete Output')


                    plt.subplot(1, 17, 12)
                    plt.imshow(out6, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Continues Output')

                    # Plotte das Vorhersagebild auf der rechten Seite
                    plt.subplot(1, 17, 13)
                    plt.imshow(round_out6, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Discrete Output')

                    plt.subplot(1, 17, 14)
                    plt.imshow(out7, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Continues Output')

                    # Plotte das Vorhersagebild auf der rechten Seite
                    plt.subplot(1, 17, 15)
                    plt.imshow(round_out7, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Discrete Output')

                    plt.subplot(1, 17, 16)
                    plt.imshow(out5, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Continues Output')

                    #Plotte das Vorhersagebild auf der rechten Seite
                    plt.subplot(1, 17, 17)
                    plt.imshow(round_out5, cmap='gray', vmin=min_val, vmax=max_val)
                    plt.title('Discrete Output')


                    # Zeige den Plot an
                    plt.savefig(f"{path}/{small}_{big}/picture_{k}")
                    #plt.show()
                pass



                    
                

        L32_result = pickle.load(
            open(
            f'/data/lspatscheck/test_samples/test_data32.pickle',
            mode = 'rb'
            )
        )
        mag32 =  np.array(L32_result['L=32 magnetizations']) [800000:]


        mean_mag_original,err_mag_original,_ =  autocorr.calc_error(np.abs(mag32))

        # Erstelle ein Dictionary, um die Ergebnisse zu speichern
        mean_mag_output = {}
        err_mag_output = {}

        mean_mag_round_output = {}
        err_mag_round_output = {}

        for i,data in enumerate([mag1,mag2,mag3,mag4,mag5,mag6,mag7]):

            mean_mag_output[i+1], err_mag_output[i+1], _ = autocorr.calc_error(np.abs(data))
        print(mag1_round)
        for i,data in enumerate([mag1_round,mag2_round,mag3_round,mag4_round,mag5_round,mag6_round,mag7_round]):
            print(np.mean(data))
            mean_mag_round_output[i+1],err_mag_round_output[i+1],_ =  autocorr.calc_error(np.abs(data))
        # Zugriff auf die Ergebnisse


        print( " Mean mag error:", (mean_mag_original - mean_mag_output[1])/mean_mag_original )

        print( " Mean mag rounded error:", (mean_mag_original- mean_mag_round_output[1])/mean_mag_original )

        print("32:",mean_mag_output[1],mean_mag_round_output[1], mean_mag_original)
        print("64:",mean_mag_output[2],mean_mag_round_output[2])
        print("128:",mean_mag_output[3],mean_mag_round_output[3])
        print("256:",mean_mag_output[4],mean_mag_round_output[4])
 

    
        mags1 = dict( output = mag1, rounded_output = mag1_round)

        mags2 = dict(output = mag2, rounded_output = mag2_round)

        mags3 = dict(output = mag3, rounded_output = mag3_round)

        mags4 = dict( output = mag4, rounded_output = mag4_round)

        mags5 = dict( output = mag5, rounded_output = mag5_round)

        mags6 = dict( output = mag6, rounded_output = mag6_round)

        mags7 = dict( output = mag7, rounded_output = mag7_round)












        pickle.dump(
            mags1,
            open(
                f"{path}/{small}_{big}/magnetizations_big'n.pickle",
                mode = 'wb'
            )
        )

        pickle.dump(
            mags2,
            open(
                f"{path}/{small}_{big*2}/magnetizations_big'n.pickle",
                mode = 'wb'
            )
        )

        pickle.dump(
            mags3,
            open(
                f"{path}/{small}_{big*2**2}/magnetizations_big'n.pickle",
                mode = 'wb'
            )
        )

        pickle.dump(
            mags4,
            open(
                f"{path}/{small}_{big*2**3}/magnetizations_big'n.pickle",
                mode = 'wb'
            )
        )

        pickle.dump(
            mags5,
            open(
                f"{path}/{small}_{big*2**4}/magnetizations_big'n.pickle",
                mode = 'wb'
            )
        )

        pickle.dump(
            mags6,
            open(
                f"{path}/{small}_{big*2**5}/magnetizations_big'n.pickle",
                mode = 'wb'
            )
        )



        pickle.dump(
            mags7,
            open(
                f"{path}/{small}_{big*2**6}/magnetizations_big'n.pickle",
                mode = 'wb'
            )
        )

