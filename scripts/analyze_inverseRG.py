


def analysis():

    length = 16
    betaJ = 0.44
    time_to_equilibrium = int(3e5)
    path_model = '/data/lspatscheck/complexUNet20000/run_10/test'

    def mag_re(m_beta0,e_beta0,beta0,beta):
        m_re = np.empty_like(beta)
        for i in range(len(beta)):
            m_re[i] = np.sum(m_beta0 * np.exp(-(beta[i]-beta0) * e_beta0 ))  / np.sum(np.exp(-(beta[i] -beta0) *e_beta0 ))
                                                                                    
        return m_re
    
    L16re_result = pickle.load(
        open(
        f'/data/lspatscheck/test_samples/test_data16.pickle',
        mode = 'rb'
        )
    )

    mag16=  np.array((L16re_result['L=16 magnetizations']))
    energy16 =  np.array((L16re_result['L=16 energies']))

    sample_size = 10000
    np.random.seed(55)
    energy16re = np.random.permutation(energy16)[:sample_size] 
    np.random.seed(55)
    mag16re = np.random.permutation(mag16)[:sample_size]

    magnetization_mean_L16re,magnetization_error_L16re,time16re = \
        autocorr.calc_error(np.abs(mag16re))
    
    print("Done")

    L16_result = pickle.load(
        gzip.open(
        f'/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size16/betaJ{betaJ}/final_result/data_2e6.gz',
        mode = 'rb'
        )
    )
    print("Done")
    energy16 = np.array((L16_result['energies']))
    mag16 = np.array((L16_result['magnetizations']))

    magnetization_mean_L16,magnetization_error_L16,time16 = \
        autocorr.calc_error(np.abs(mag16))
    


    L32_result = pickle.load(
        gzip.open(
        f'/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size32/betaJ{betaJ}/final_result/data_2e6.gz',
        mode = 'rb'
        )
    )
    print("Done")
    energy32 = np.array((L32_result['energies']))


    mag32 = np.array((L32_result['magnetizations']))
    print(np.mean(np.abs(mag32)),np.mean(np.abs(mag32[int(1e5):])),np.mean(np.abs(mag32[int(5e5):])),np.mean(np.abs(mag32[int(1e6):]) ))

    magnetization_mean_L32,magnetization_error_L32,time32 = \
        autocorr.calc_error(np.abs(mag32))

    L64_result = pickle.load(
        gzip.open(
        f'/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size64/betaJ{betaJ}/final_result/data_2e6.gz',
        mode = 'rb'
        )
    )

    energy64 = np.array((L64_result['energies']))
    mag64 = np.array((L64_result['magnetizations']))

    magnetization_mean_L64,magnetization_error_L64,time64 = \
        autocorr.calc_error(np.abs(mag64))
    

    print("Done")
    L128_result = pickle.load(
        gzip.open(
        f'/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size128/betaJ{betaJ}/final_result/data_2e6.gz',
        mode = 'rb'
        )
    )

    energy128 = np.array((L128_result['energies']))
    mag128 = np.array((L128_result['magnetizations']))

    magnetization_mean_L128,magnetization_error_L128,time128 = \
        autocorr.calc_error(np.abs(mag128))
    
    L256_result = pickle.load(
        open(
        f'/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size256/betaJ{betaJ}/final_result/simple_data_2e6.pickle',
        mode = 'rb'
        )
    )

    energy256 = np.array((L256_result['energies']))
    mag256 = np.array((L256_result['magnetizations']))

    magnetization_mean_L256,magnetization_error_L256,time256 = \
        autocorr.calc_error(np.abs(mag256))
    
    """    
    L256_result = pickle.load(
        gzip.open(
        f'/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size256/betaJ{betaJ}/final_result/data_2e6.gz',
        mode = 'rb'
        )
    )
    print("Done")
    energy256 = np.array((L256_result['energies']))
    mag256 = np.array((L256_result['magnetizations']))

    magnetization_mean_L256,magnetization_error_L256,time256 = \
        autocorr.calc_error(np.abs(mag256))
    

    L512_result = pickle.load(
        gzip.open(
        f'/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size512/betaJ{betaJ}/final_result/data_2e6.gz',
        mode = 'rb'
        )
    )
    print("Done")
    energy512 = np.array((L512_result['energies']))
    mag512 = np.array((L512_result['magnetizations']))

    magnetization_mean_L512,magnetization_error_L512,time512 = \
        autocorr.calc_error(np.abs(mag512))
    """  

    print("loaded sim data")

    # Initialize lists to store the results for each path_model

    all_magres = []

    all_magre_means = []

    all_magre_errors = []

    all_magres_sq = []


    beta = np.linspace(0.4400,0.4410,200)


    #mag512weight = mag_re(np.abs(mag512),energy512,betaJ,beta)

    mag256weight = mag_re(np.abs(mag256),energy256,betaJ,beta)

    mag128weight = mag_re(np.abs(mag128),energy128,betaJ,beta)

    mag64weight = mag_re(np.abs(mag64),energy64,betaJ,beta)

    mag32weight = mag_re(np.abs(mag32),energy32,betaJ,beta)

    mag16weight = mag_re(np.abs(mag16),energy16,betaJ,beta)

    mag = [mag256weight,mag128weight,mag64weight,mag32weight,mag16weight]
    mag_error = [magnetization_error_L256,magnetization_error_L128,magnetization_error_L64,magnetization_error_L32,magnetization_error_L16]
    mag_mean = [magnetization_mean_L256,magnetization_mean_L128,magnetization_mean_L64,magnetization_mean_L32,magnetization_mean_L16]

    path_models = ['/data/lspatscheck/complexUNet2000/run_10/test','/data/lspatscheck/complexUNet2000/run_11/test','/data/lspatscheck/complexUNet2000/run_12/test','/data/lspatscheck/complexUNet2000/run_15/test','/data/lspatscheck/complexUNet2000/run_16/test','/data/lspatscheck/complexUNet2000/run_17/test','/data/lspatscheck/complexUNet2000/run_18/test','/data/lspatscheck/complexUNet2000/run_19/test','/data/lspatscheck/complexUNet2000/run_20/test']

    # Function to load data and calculate magnetization and errors
    def load_and_process_data(path_model):
        mag_data = {}
        mag_sizes = [2048,1024, 512, 256, 128, 64, 32]
        
        for size in mag_sizes:
            with open(f"{path_model}/16_{size}/magnetizations_big'n.pickle", "rb") as file:
                data = pickle.load(file)
                mag_data[size] = np.array(data['rounded_output'])
        
        print(f"Data loaded for {path_model}")
        
        magnetization_errors = []
        magnetization_means = []
        
        for size in mag_sizes:
            mean, error, time = autocorr.calc_error(np.abs(mag_data[size]))
            magnetization_means.append(mean)
            magnetization_errors.append(error)  

            print(f"{size}",mean,error,time)
        
        mag_reweight = []
        mag_sq_reweight = []
        
        for size in mag_sizes:
            mag_reweight.append(mag_re(np.abs(mag_data[size]), energy16re, betaJ, beta))  # Assuming energy16re is used for reweighting
            mag_sq_reweight.append(mag_re(np.abs((mag_data[size]*size**2)**2), energy16re, betaJ, beta))

        
        return mag_reweight,magnetization_means, magnetization_errors ,mag_sq_reweight

    # Function to calculate critical renormalization values
    def calculate_critical_renorm(mag_reweights,mag):
        crit_renorms = []
        mag16 = mag[-1]  # Assuming the smallest size is at the end
        dmag16 =  spip.interp1d(beta,np.gradient(mag16,beta))
        mag16 = spip.interp1d(beta,mag16)
        beta_c = 0.5 * np.log(1 + np.sqrt(2)) 
        beta_c = 0.4404728

        mag_interp = {size: spip.interp1d(beta, mag_reweights[i]) for i, size in enumerate([2048,1024, 512, 256, 128, 64, 32])}
        mag_interp[16] = mag16

        dmag_interp = {size: spip.interp1d(beta, np.gradient(mag_reweights[i],beta)) for i, size in enumerate([2048,1024, 512, 256, 128, 64, 32])}
        dmag_interp[16] = dmag16
        
        for i, size1 in enumerate([16,32, 64, 128, 256, 512,1024]):
            for j, size2 in enumerate([32,64, 128, 256, 512, 1024,2048][i:]):



                crit_renorms.append(
                    -np.log(dmag_interp[size2](beta_c) / dmag_interp[size1](beta_c)) / (np.log(2) * (j + 1))
                )
        
        return crit_renorms

    # Iterate over each path_model and process the data
    all_data1 = []
    all_data2 = []
    all_data3 = []

    for path_model in path_models:
        magre, magre_mean, magre_error,magsqre = load_and_process_data(path_model)
        all_magres.append(magre)
        all_magre_means.append(magre_mean)
        all_magre_errors.append(magre_error)
        all_magres_sq.append(magsqre)

        crit_renorms = calculate_critical_renorm(magre,mag)

        data1 = crit_renorms[:7]
        list = [6,12,17,21,14,26,27]
        data2 = [crit_renorms[i] for i in list]
        list = [0,7,13,18,22,25,27]
        data3 = [crit_renorms[i] for i in list]

        all_data1.append(data1)
        all_data2.append(data2)
        all_data3.append(data3)



    print(np.mean(crit_renorms))

    all_data1 = np.array(all_data1)
    all_data2 = np.array(all_data2)
    all_data3 = np.array(all_data3)
    print(all_data3)
    data = dict( up16 = all_data1 , single_up = all_data3   )

    pickle.dump(
    data,
    open(
        f'/tikhome/lspatscheck/Documents/bsc/simulation_data/inverse_renorm/beta_crits.pickle',
        mode = 'wb'
        )
    )
    # Create figures with error bars
    """
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    print(np.arange(0,len(all_data1[1])), all_data1[1],all_data1[1],all_data1[2])
    ax[0].plot(np.arange(0,len(all_data1[0])),all_data1[0],'ro' ,label='Data Set 1')
    ax[0].plot(np.arange(0,len(all_data1[1])),all_data1[1],'bo' )
    ax[0].plot(np.arange(0,len(all_data1[2])),all_data1[2],'go' )
    ax[0].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    #ax[0].set_yscale('log')
    ax[0].set_title('Deviation $\\beta/\\nu$ - Set 1: 16up', fontsize=16)
    ax[0].set_xlabel('Data Point', fontsize=14)
    ax[0].set_ylabel('Deviation', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=12)

    ax[1].plot(np.arange(0,len(all_data2[0])),all_data2[0],'ro' ,label='Data Set 2')
    ax[1].plot(np.arange(0,len(all_data2[1])),all_data2[1],'bo' )
    ax[1].plot(np.arange(0,len(all_data2[2])),all_data2[2],'go' )
    ax[1].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    #ax[1].set_yscale('log')
    ax[1].set_title('Deviation $\\beta/\\nu$ - Set 2: 2048down', fontsize=16)
    ax[1].set_xlabel('Data Point', fontsize=14)
    ax[1].set_ylabel('Deviation', fontsize=14)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(fontsize=12)

    ax[2].plot(np.arange(0,len(all_data1[0])),all_data3[0],'ro' ,label='Data Set 3')
    ax[2].plot(np.arange(0,len(all_data1[1])),all_data3[1],'bo' )
    ax[2].plot(np.arange(0,len(all_data1[2])),all_data3[2],'go' )
    ax[2].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    #ax[2].set_yscale('log')
    ax[2].set_title('Deviation $\\beta/\\nu$ - Set 3: (j-i) = 1', fontsize=16)
    ax[2].set_xlabel('Data Point', fontsize=14)
    ax[2].set_ylabel('Deviation', fontsize=14)
    ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[2].legend(fontsize=12)

    plt.tight_layout()
    #plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/test_inverse_complex_UNet/inverse_crits_abs.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    print(np.arange(0,len(all_data1[1])), all_data1[1],all_data1[1],all_data1[2])
    ax[0].plot(np.arange(0,len(all_data1[0]))+1, (all_data1[0]-0.125)/0.125,'ro', label='run 1')
    ax[0].plot(np.arange(0,len(all_data1[0]))+1, (all_data1[1]-0.125)/0.125,'bo',label='run 2')
    ax[0].plot(np.arange(0,len(all_data1[0]))+1, (all_data1[2]-0.125)/0.125,'go',label='run 3')
    #ax[0].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    #ax[0].set_yscale('log')
    ax[0].set_title('Relative Deviation $\\beta/\\nu$  - Set 1: $L_{i=0} = 16$', fontsize=16)
    ax[0].set_xlabel('Data Point $j$', fontsize=14)
    ax[0].set_ylabel('Relative Deviation', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=12)

    ax[1].plot(np.arange(0,len(all_data1[0])), (all_data2[0]-0.125)/0.125,'ro', label='run 1')
    ax[1].plot(np.arange(0,len(all_data1[0])), (all_data2[1]-0.125)/0.125,'bo',label='run 2')
    ax[1].plot(np.arange(0,len(all_data1[0])), (all_data2[2]-0.125)/0.125,'go',label='run 3')
    #ax[1].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    #ax[1].set_yscale('log')
    ax[1].set_title('Relative Deviation $\\beta/\\nu$ - Set 2: $L_{j=7} = 2048$', fontsize=16)
    ax[1].set_xlabel('Data Point $i$', fontsize=14)
    ax[1].set_ylabel('Relative Deviation', fontsize=14)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(fontsize=12)

    ax[2].plot(np.arange(0,len(all_data1[0])), (all_data3[0]-0.125)/0.125,'ro', label='run 1')
    ax[2].plot(np.arange(0,len(all_data1[0])), (all_data3[1]-0.125)/0.125,'bo',label='run 2')
    ax[2].plot(np.arange(0,len(all_data1[0])), (all_data3[2]-0.125)/0.125,'go',label='run 3')
    #ax[2].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    #ax[2].set_yscale('log')
    ax[2].set_title('Relative Deviation $\\beta/\\nu$ - Set 3: $(j-i) = 1$', fontsize=16)
    ax[2].set_xlabel('Data Point $i$', fontsize=14)
    ax[2].set_ylabel('Relative Deviation', fontsize=14)
    ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[2].legend(fontsize=12)

    plt.tight_layout()
    #plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/test_inverse_complex_UNet/inverse_crits_rela.png', dpi=300, bbox_inches='tight')
    plt.show()

    stacked_data1 = np.vstack( (all_data1[1], all_data1[2]))
    # Berechnen des Mittelwerts zu jedem Zeitpunkt (entlang der ersten Achse)
    mean_data1 = np.mean(stacked_data1, axis=0)
    err1 = np.std(stacked_data1,axis= 0)/ np.sqrt(3)

    stacked_data2 = np.vstack( (all_data2[1], all_data2[2]))
    # Berechnen des Mittelwerts zu jedem Zeitpunkt (entlang der ersten Achse)
    mean_data2= np.mean(stacked_data2, axis=0)
    err2 = np.std(stacked_data2,axis= 0)/ np.sqrt(3)

    stacked_data3 = np.vstack( (all_data3[1], all_data3[2]))
    # Berechnen des Mittelwerts zu jedem Zeitpunkt (entlang der ersten Achse)
    mean_data3 = np.mean(stacked_data3, axis=0)
    err3= np.std(stacked_data3,axis= 0)/ np.sqrt(3)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].plot(np.arange(0,len(all_data1[0]))+1, (mean_data1-0.125)/0.125,'ro', label='Mean')
    ax[0].fill_between(np.arange(len(mean_data1))+1, (mean_data1 +err1-0.125)/0.125 , (mean_data1 - err1 - 0.125)/0.125, color='red', alpha=0.2)
    #ax[0].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    #ax[0].set_yscale('log')
    ax[0].set_title('Relative Deviation $\\beta/\\nu$  - Set 1: $L_{i=0} = 16$', fontsize=16)
    ax[0].set_xlabel('Data Point $j$', fontsize=14)
    ax[0].set_ylabel('Mean Relative Deviation', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=12)

    ax[1].plot(np.arange(len(all_data1[0])), (mean_data2-0.125)/0.125,'ro', label='Mean')
    ax[1].fill_between(np.arange(len(mean_data1)), (mean_data2 +err2-0.125)/0.125 , (mean_data2 - err2 - 0.125)/0.125, color='red', alpha=0.2)
    #ax[1].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    #ax[1].set_yscale('log')
    ax[1].set_title('Relative Deviation $\\beta/\\nu$ - Set 2: $L_{j=7} = 2048$', fontsize=16)
    ax[1].set_xlabel('Data Point $i$', fontsize=14)
    ax[1].set_ylabel('Mean Relative Deviation', fontsize=14)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(fontsize=12)

    ax[2].plot(np.arange(0,len(all_data3[0])), (mean_data3-0.125)/0.125,'ro', label='Mean')
    ax[2].fill_between(np.arange(len(mean_data3)), (mean_data3 +err3-0.125)/0.125 , (mean_data3 - err3 - 0.125)/0.125, color='red', alpha=0.2)
    #ax[2].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    #ax[2].set_yscale('log')
    ax[2].set_title('Relative Deviation $\\beta/\\nu$ - Set 3: $(j-i) = 1$', fontsize=16)
    ax[2].set_xlabel('Data Point $i$', fontsize=14)
    ax[2].set_ylabel('Mean Relative Deviation', fontsize=14)
    ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[2].legend(fontsize=12)

    plt.tight_layout()
    #plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/test_inverse_complex_UNet/inverse_crits_rela_mean_corr.png', dpi=300, bbox_inches='tight')
    plt.show()
    """


    L = [256,128,64,32]
# Einstellungen für einheitliche Schriftarten
    plt.rcParams.update({'font.size': 24, 'font.family': 'serif'})

    # Plotten


    for i in range(len(L)):
        fig, ax = plt.subplots(figsize=(12, 8))

        stacked_mag = np.vstack(( all_magres[0][i+3],all_magres[1][i+3], all_magres[2][i+3],all_magres[3][i+3],all_magres[4][i+3], all_magres[5][i+3],all_magres[7][i+3],all_magres[8][i+3]))
        # Berechnen des Mittelwerts zu jedem Zeitpunkt (entlang der ersten Achse)
        mean_mag = np.mean(stacked_mag, axis=0)
        err= np.std(stacked_mag,axis= 0)/ np.sqrt(9)


        # Originale Magnetisierung mit Fehlerbalken
        ax.plot(beta, mag[i], "r-", label=f"L={L[i]}", linewidth=2)
        ax.fill_between(beta, mag[i] - mag_error[i], mag[i] + mag_error[i], color='r', alpha=0.2)

        # Renormierte Magnetisierung mit Fehlerbalken
        ax.plot(beta, mean_mag, "b-.", label=f"L'={L[i]}", linewidth=2)
        
        ax.fill_between(beta, mean_mag + err , mean_mag - err , color='g', alpha=0.8)
        ax.fill_between(beta, mean_mag + np.mean(all_magre_errors[:][i+3]) , mean_mag - np.mean(all_magre_errors[:][i+3]) , color='b', alpha=0.2)

        # Achsenbeschriftungen und Titel
        ax.set_xlabel('$\\beta$', fontsize=28)
        ax.set_ylabel('$\\langle \\left|m \\right| \\rangle$', fontsize=28)
        #ax.set_title(f'Reweighted Magnetization', fontsize=20, pad=20)

        # Legende
        ax.legend(fontsize=28)

        plt.xticks([0.4400,0.4406,0.4410], ["$0.4400$","$0.4406$","$0.4410$"], fontsize=24)
        # Achsenskalierung verbessern
        ax.set_xlim([0.4400, 0.4410])
        print([min([m.min() for m in mag[i] ]), max([m.max() for m in mag[i] ]) ])

        ax.set_ylim([min([m.min() for m in mag[i]-0.002 ]), max([m.max() for m in mag[i]+0.002 ]) ])

        #ax.set_ylim([0.50,0.75 ])
        # Layout verbessern
        plt.tight_layout()

        # Speichern der Abbildung
        plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/test_inverse_complex_UNet/inverse_group_flow_UNet20000_{L[i]}.pdf', dpi=300, bbox_inches='tight')

        # Zeigen der Abbildung
        plt.show()
    """

    dmag16 =  spip.interp1d(beta,mag16weight)

    dmag512re = spip.interp1d(beta,mag512re)
    dmag256re = spip.interp1d(beta,mag256re)
    dmag128re = spip.interp1d(beta,mag128re)
    dmag64re = spip.interp1d(beta,mag64re)
    dmag32re = spip.interp1d(beta,mag32re)
    dmag16re = spip.interp1d(beta,mag16re)

    beta_c = 0.5 * np.log(1 + np.sqrt(2)) 


    crit_renorm16_32 = -np.log(dmag32re(beta_c)/dmag16(beta_c)) /(np.log(2)*(1))
    crit_renorm16_64 = -np.log(dmag64re(beta_c)/dmag16(beta_c)) /(np.log(2)*(2))
    crit_renorm16_128 = -np.log(dmag128re(beta_c)/dmag16(beta_c)) /(np.log(2)*(3))
    crit_renorm16_256 = -np.log(dmag256re(beta_c)/dmag16(beta_c)) /(np.log(2)*(4))
    crit_renorm16_512 = -np.log(dmag512re(beta_c)/dmag16(beta_c)) /(np.log(2)*(5))

    crit_renorm32_64 = -np.log(dmag64re(beta_c)/dmag32re(beta_c)) /(np.log(2)*(1))
    crit_renorm32_128 = -np.log(dmag128re(beta_c)/dmag32re(beta_c)) /(np.log(2)*(2))
    crit_renorm32_256 = -np.log(dmag256re(beta_c)/dmag32re(beta_c)) /(np.log(2)*(3))
    crit_renorm32_512 = -np.log(dmag512re(beta_c)/dmag32re(beta_c)) /(np.log(2)*(4))

    crit_renorm64_128 = -np.log(dmag128re(beta_c)/dmag64re(beta_c)) /(np.log(2)*(1))
    crit_renorm64_256 = -np.log(dmag256re(beta_c)/dmag64re(beta_c)) /(np.log(2)*(2))
    crit_renorm64_512 = -np.log(dmag512re(beta_c)/dmag64re(beta_c)) /(np.log(2)*(3))

    crit_renorm128_256 = -np.log(dmag256re(beta_c)/dmag128re(beta_c)) /(np.log(2)*(1))
    crit_renorm128_512 = -np.log(dmag512re(beta_c)/dmag128re(beta_c)) /(np.log(2)*(2))

    crit_renorm256_512 = -np.log(dmag512re(beta_c)/dmag256re(beta_c)) /(np.log(2)*(1))


    error_percent = 0.05

    print(crit_renorm128_64,crit_renorm128_32,crit_renorm128_16,crit_renorm64_32,crit_renorm64_16)

    data1 = np.array([crit_renorm16_512, crit_renorm32_512, crit_renorm64_512, crit_renorm128_512, crit_renorm256_512])
    data2 = np.array([crit_renorm16_32, crit_renorm32_64, crit_renorm64_128, crit_renorm128_256, crit_renorm256_512])
    data3 = np.array([crit_renorm16_32, crit_renorm16_64, crit_renorm16_128,crit_renorm16_256,crit_renorm16_256,crit_renorm16_512])


    # Calculate relative deviations
    relative_deviation1 = (data1 - 0.125) / 0.125
    relative_deviation2 = (data2 - 0.125) / 0.125
    relative_deviation3 = (data3 - 0.125) / 0.125



    # Create figures with error bars
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].errorbar(np.arange(0,len(relative_deviation1)), relative_deviation1, yerr=[-relative_down1,relative_dup1],  fmt='o', capsize=8, elinewidth=2, label='Data Set 1')
    ax[0].set_title('Relative Deviation with Error Bars - Set 1', fontsize=16)
    ax[0].set_xlabel('Data Point', fontsize=14)
    ax[0].set_ylabel('Relative Deviation', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=12)

    ax[1].errorbar(np.arange(0,len(relative_deviation2)), relative_deviation2, yerr=[np.array([-1,1,-1])*relative_down2,relative_dup2], fmt='s', capsize=8, elinewidth=2, label='Data Set 2')
    ax[1].set_title('Relative Deviation with Error Bars - Set 2', fontsize=16)
    ax[1].set_xlabel('Data Point', fontsize=14)
    ax[1].set_ylabel('Relative Deviation', fontsize=14)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(fontsize=12)

    ax[2].errorbar(np.arange(0,len(relative_deviation3)), relative_deviation3, yerr=[-relative_down3,relative_dup3], fmt='^', capsize=8, elinewidth=2, label='Data Set 3')
    #ax[2].set_yscale('log')
    ax[2].set_title('Relative Deviation with Error Bars - Set 3', fontsize=16)
    ax[2].set_xlabel('Data Point', fontsize=14)
    ax[2].set_ylabel('Relative Deviation', fontsize=14)
    ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[2].legend(fontsize=12)

    plt.tight_layout()
    ##plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/forward_crits_rela.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Create figures with error bars
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].errorbar(np.arange(0,len(data1)), data1, yerr=[-err_down1,err_up1],  fmt='o', capsize=8, elinewidth=2, label='Data Set 1')
    ax[0].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    ax[0].set_title('Relative Deviation with Error Bars - Set 1', fontsize=16)
    ax[0].set_xlabel('Data Point', fontsize=14)
    ax[0].set_ylabel('Relative Deviation', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=12)

    ax[1].errorbar(np.arange(0,len(data2)), data2, yerr=[np.array([-1,1,-1])*err_down2,err_up2], fmt='s', capsize=8, elinewidth=2, label='Data Set 2')
    ax[1].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    ax[1].set_title('Relative Deviation with Error Bars - Set 2', fontsize=16)
    ax[1].set_xlabel('Data Point', fontsize=14)
    ax[1].set_ylabel('Relative Deviation', fontsize=14)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(fontsize=12)

    ax[2].errorbar(np.arange(0,len(data3)), data3, yerr=[-err_down3,err_up3], fmt='^', capsize=8, elinewidth=2, label='Data Set 3')
    ax[2].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    ax[2].set_title('Relative Deviation with Error Bars - Set 3', fontsize=16)
    ax[2].set_xlabel('Data Point', fontsize=14)
    ax[2].set_ylabel('Relative Deviation', fontsize=14)
    ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[2].legend(fontsize=12)

    plt.tight_layout()
    ##plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/forward_crits_abs.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    L = [512,256,128,64,32]
# Einstellungen für einheitliche Schriftarten
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

    # Plotten
    for i in range(len(L)):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Originale Magnetisierung mit Fehlerbalken
        ax.plot(beta, mag[i], "r-", label=f"L={L[i]}", linewidth=2)
        ax.fill_between(beta, mag[i] - mag_error[i], mag[i] + mag_error[i], color='r', alpha=0.2)

        # Renormierte Magnetisierung mit Fehlerbalken
        ax.plot(beta, magre[i], "b-.", label=f"L'={L[i]}", linewidth=2)
        ax.fill_between(beta, magre[i] - magre_error[i], magre[i] + magre_error[i], color='b', alpha=0.2)

        # Achsenbeschriftungen und Titel
        ax.set_xlabel(r'$\beta$', fontsize=16)
        ax.set_ylabel(r'$|m|$', fontsize=16)
        ax.set_title(f'Reweighted Magnetization at $\\beta_c = {betaJ}$', fontsize=18, pad=20)

        # Legende
        ax.legend(fontsize=14)

        # Raster hinzufügen
        ax.grid(True, linestyle='--', alpha=0.5)

        # Achsenskalierung verbessern
        ax.set_xlim([beta.min(), beta.max()])
        ax.set_ylim([min([m.min() for m in mag + magre]) - 0.02, max([m.max() for m in mag + magre]) + 0.02])

        # Layout verbessern
        plt.tight_layout()

        # Speichern der Abbildung
        #plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/group_flow_{i}.pdf', dpi=300, bbox_inches='tight')

        # Zeigen der Abbildung
        plt.show()
    
    """

    mag_sq128weight = mag_re(np.abs(mag128*128**2)**2,energy128,betaJ,beta)

    mag_sq64weight = mag_re(np.abs(mag64*64**2)**2,energy64,betaJ,beta)

    mag_sq32weight = mag_re(np.abs(mag32*32**2)**2,energy32,betaJ,beta)

    mag_sq16weight = mag_re(np.abs(mag16*16**2)**2,energy16,betaJ,beta)

    susz128weigth = (1/128**2) * ( mag_sq128weight - (mag128weight*128**2)**2   )

    susz64weigth = (1/64**2) * ( mag_sq64weight - (mag64weight*64**2)**2   )

    susz32weigth = (1/32**2) * ( mag_sq32weight - (mag32weight*32**2)**2   )

    susz16weigth = (1/16**2) * ( mag_sq16weight - (mag16weight*16**2)**2   )

    all_suszre = []
    mag_sizes = [2048,1024, 512, 256, 128, 64, 32] * 9
    for i in range(len(mag_sizes)):
        #print(type(mag_sizes[i]),type(all_magres[i]))
        all_suszre.append( 1/(mag_sizes[i]**2) * (all_magres_sq[i//7][i%7] - (all_magres[i//7][i%7]*mag_sizes[i]**2)**2 ))


    mag_sq16re = mag_re(np.abs((mag16re*16**2)**2),energy16re,betaJ,beta)



    susz16re = (1/16**2) *( mag_sq16re - (mag16weight*16**2)**2    )


    
    susz_mean_L64re,susz_error_L64re,time64re = \
        autocorr.calc_error(susz16re)
    
    susz_mean_L16,susz_error_L16,time16 = \
        autocorr.calc_error(susz16weigth)
    
    susz_mean_L32,susz_error_L32,time32 = \
        autocorr.calc_error(susz32weigth)
    
    susz_mean_L64,susz_error_L64,time64 = \
        autocorr.calc_error(susz64weigth)
    
    susz_mean_L128,susz_error_L128,time128 = \
        autocorr.calc_error(susz128weigth)


    susz = [susz128weigth,susz64weigth,susz32weigth,susz16weigth]
    #suszre = [susz64re,susz32re,susz16re]
    susz_mean= [susz_mean_L128,susz_mean_L64,susz_mean_L32,susz_mean_L16]
    susz_error = [susz_error_L128,susz_error_L64,susz_error_L32,susz_error_L16]
    #suszre_error = [susz_error_L64re,susz_error_L32re,susz_error_L16re]

        # Function to calculate critical renormalization values
    def calculate_critical_renorm(susz_reweights,susz16):
        crit_renorms = []
        mag16 = susz16  # Assuming the smallest size is at the end
        dmag16 =  spip.interp1d(beta,np.gradient(mag16,beta))
        beta_c = 0.5 * np.log(1 + np.sqrt(2)) 
        beta_c = 0.4404728

        mag_interp = {size: spip.interp1d(beta, np.gradient(susz_reweights[i],beta)) for i, size in enumerate([2048,1024, 512, 256, 128, 64, 32])}
        mag_interp[16] = dmag16
        
        for i, size1 in enumerate([16,32, 64, 128, 256, 512,1024]):
            for j, size2 in enumerate([32,64, 128, 256, 512, 1024,2048][i:]):
                crit_renorms.append(
                    np.log(mag_interp[size2](beta_c) / mag_interp[size1](beta_c)) / (np.log(2) * (j + 1))
                )
                if size1 ==32:
                    print(np.log(mag_interp[size2](beta_c) / mag_interp[size1](beta_c)) / (np.log(2) * (j + 1)))
                #print(np.log(mag_interp[size2](beta_c) / mag_interp[size1](beta_c)) / (np.log(2) * (j + 1)))
        return crit_renorms

    all_data1 = []
    all_data2 = []
    all_data3 = []

    for i in range(len(path_models)):

        crit_renorms = calculate_critical_renorm(all_suszre[i*7:i*7+7],susz16re)

        data1 = crit_renorms[:7]

        list = [6,12,17,21,14,26,27]
        data2 = [crit_renorms[i] for i in list]
        list = [0,7,13,18,22,25,27]
        data3 = [crit_renorms[i] for i in list]

        all_data1.append(data1)
        all_data2.append(data2)
        all_data3.append(data3)
    print(np.mean(crit_renorms))

    all_data1 = np.array(all_data1)
    all_data2 = np.array(all_data2)
    all_data3 = np.array(all_data3)
    print(all_data3)
    data = dict( up16 = all_data1 , single_up = all_data3   )

    pickle.dump(
    data,
    open(
        f'/tikhome/lspatscheck/Documents/bsc/simulation_data/inverse_renorm/gamma_crits.pickle',
        mode = 'wb'
        )
    )
    """
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    print(np.arange(0,len(all_data1[1])), all_data1[1],all_data1[1],all_data1[2])
    ax[0].plot(np.arange(0,len(all_data1[0])),all_data1[0],'ro' ,label='Data Set 1')
    ax[0].plot(np.arange(0,len(all_data1[1])),all_data1[1],'bo' )
    ax[0].plot(np.arange(0,len(all_data1[2])),all_data1[2],'go' )
    ax[0].axhline(y=1.75, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\gamma/\\nu$")
    ax[0].set_yscale('log')
    ax[0].set_title('Deviation $\\gamma/\\nu$ - Set 1: 16up', fontsize=16)
    ax[0].set_xlabel('Data Point', fontsize=14)
    ax[0].set_ylabel('Deviation', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=12)

    ax[1].plot(np.arange(0,len(all_data2[0])),all_data2[0],'ro' ,label='Data Set 2')
    ax[1].plot(np.arange(0,len(all_data2[1])),all_data2[1],'bo' )
    ax[1].plot(np.arange(0,len(all_data2[2])),all_data2[2],'go' )
    ax[1].axhline(y=1.75, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\gamma/\\nu$")
    ax[1].set_yscale('log')
    ax[1].set_title('Deviation $\\gamma/\\nu$ - Set 2: 1024down', fontsize=16)
    ax[1].set_xlabel('Data Point', fontsize=14)
    ax[1].set_ylabel('Deviation', fontsize=14)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(fontsize=12)

    ax[2].plot(np.arange(0,len(all_data3[0])),all_data3[0],'ro' ,label='Data Set 3')
    ax[2].plot(np.arange(0,len(all_data3[1])),all_data3[1],'bo' )
    ax[2].plot(np.arange(0,len(all_data3[2])),all_data3[2],'go' )
    ax[2].axhline(y=1.75, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\gamma/\\nu$")
    ax[2].set_yscale('log')
    ax[2].set_title('Deviation $\\gamma/\\nu$ - Set 3: (j-i) = 1', fontsize=16)
    ax[2].set_xlabel('Data Point', fontsize=14)
    ax[2].set_ylabel('Deviation', fontsize=14)
    ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[2].legend(fontsize=12)

    plt.tight_layout()
    #plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/test_inverse_complex_UNet/inverse_crits_susz_abs.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    print(np.arange(0,len(all_data1[1])), all_data1[1],all_data1[1],all_data1[2])
    ax[0].plot(np.arange(0,len(all_data1[0]))+1, (all_data1[0]-1.75)/1.75,'ro', label='run 1')
    ax[0].plot(np.arange(0,len(all_data1[0]))+1, (all_data1[1]-1.75)/1.75,'bo',label='run 2')
    ax[0].plot(np.arange(0,len(all_data1[0]))+1, (all_data1[2]-1.75)/1.75,'go',label='run 3')
    #ax[0].axhline(y=1.75, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    #ax[0].set_yscale('log')
    ax[0].set_title('Relative Deviation $\\gamma/\\nu$  - Set 1: $L_{i=0}=16$', fontsize=16)
    ax[0].set_xlabel('Data Point $j$', fontsize=14)
    ax[0].set_ylabel('Relative Deviation', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=12)

    ax[1].plot(np.arange(0,len(all_data1[0])), (all_data2[0]-1.75)/1.75,'ro', label='run 1')
    ax[1].plot(np.arange(0,len(all_data1[0])), (all_data2[1]-1.75)/1.75,'bo',label='run 2')
    ax[1].plot(np.arange(0,len(all_data1[0])), (all_data2[2]-1.75)/1.75,'go',label='run 3')
    #ax[1].axhline(y=1.75, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\gamma/\\nu$")
    #ax[1].set_yscale('log')
    ax[1].set_title('Relative Deviation $\\gamma/\\nu$ - Set 2: $L_{j=7}= 2048$', fontsize=16)
    ax[1].set_xlabel('Data Point $i$', fontsize=14)
    ax[1].set_ylabel('Relative Deviation', fontsize=14)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(fontsize=12)

    ax[2].plot(np.arange(0,len(all_data1[0])), (all_data3[0]-1.75)/1.75,'ro', label='run 1')
    ax[2].plot(np.arange(0,len(all_data1[0])), (all_data3[1]-1.75)/1.75,'bo',label='run 2')
    ax[2].plot(np.arange(0,len(all_data1[0])), (all_data3[2]-1.75)/1.75,'go',label='run 3')
    #ax[2].axhline(y=1.75, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\gamma/\\nu$")
    #ax[2].set_yscale('log')
    ax[2].set_title('Relative Deviation $\\gamma/\\nu$ - Set 3: $(j-i) = 1$', fontsize=16)
    ax[2].set_xlabel('Data Point $i$', fontsize=14)
    ax[2].set_ylabel('Relative Deviation', fontsize=14)
    ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[2].legend(fontsize=12)

    plt.tight_layout()
    #plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/test_inverse_complex_UNet/inverse_crits_susz_rela.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    stacked_data1 = np.vstack( (all_data1[1], all_data1[2]))
    # Berechnen des Mittelwerts zu jedem Zeitpunkt (entlang der ersten Achse)
    mean_data1 = np.mean(stacked_data1, axis=0)
    err1 = np.std(stacked_data1,axis= 0)/ np.sqrt(3)
    print(err1)

    stacked_data2 = np.vstack((all_data2[1], all_data2[2]))
    # Berechnen des Mittelwerts zu jedem Zeitpunkt (entlang der ersten Achse)
    mean_data2 = np.mean(stacked_data2, axis=0)
    err2 = np.std(stacked_data2,axis= 0)/ np.sqrt(3)
    print(err2)

    stacked_data3 = np.vstack(( all_data3[1], all_data3[2]))
    # Berechnen des Mittelwerts zu jedem Zeitpunkt (entlang der ersten Achse)
    mean_data3 = np.mean(stacked_data3, axis=0)
    err3= np.std(stacked_data3,axis= 0)/ np.sqrt(3)
    print(err3)


    ax[0].plot(np.arange(0,len(all_data1[0]))+1, (mean_data1-1.75)/1.75,'ro', label='Mean')
    ax[0].fill_between(np.arange(len(mean_data1))+1, (mean_data1 +err1-1.75)/1.75 , (mean_data1 - err1 - 1.75)/1.75, color='red', alpha=0.2)

    #ax[0].axhline(y=1.75, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    #ax[0].set_yscale('log')
    ax[0].set_title('Relative Deviation $\\gamma/\\nu$  - Set 1: $L_{i=0} =16$', fontsize=16)
    ax[0].set_xlabel('Data Point $j$', fontsize=14)
    ax[0].set_ylabel('Relative Deviation', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=12)

    ax[1].plot(np.arange(0,len(all_data2[0])), (mean_data2-1.75)/1.75,'ro', label='Mean')
    ax[1].fill_between(range(len(mean_data2)), (mean_data2 +err2-1.75)/1.75 , (mean_data2 - err2 - 1.75)/1.75, color='red', alpha=0.2)
    #ax[1].axhline(y=1.75, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\gamma/\\nu$")
    #ax[1].set_yscale('log')
    ax[1].set_title('Relative Deviation $\\gamma/\\nu$ - Set 2: $L_{j=7} = 2048$', fontsize=16)
    ax[1].set_xlabel('Data Point $i$', fontsize=14)
    ax[1].set_ylabel('Relative Deviation', fontsize=14)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(fontsize=12)

    ax[2].plot(np.arange(0,len(all_data3[0])), (mean_data3-1.75)/1.75,'ro', label='Mean')
    ax[2].fill_between(range(len(mean_data3)), (mean_data3 +err3-1.75)/1.75 , (mean_data3 - err3 - 1.75)/1.75, color='red', alpha=0.2)
    #ax[2].axhline(y=1.75, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\gamma/\\nu$")
    #ax[2].set_yscale('log')
    ax[2].set_title('Relative Deviation $\\gamma/\\nu$ - Set 3: $(j-i) = 1$', fontsize=16)
    ax[2].set_xlabel('Data Point $i$', fontsize=14)
    ax[2].set_ylabel('Relative Deviation', fontsize=14)
    ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[2].legend(fontsize=12)

    plt.tight_layout()
    #plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/test_inverse_complex_UNet/inverse_crits_susz_rela_mean_corr.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    """

    L = [128,64,32]
# Einstellungen für einheitliche Schriftarten
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

    # Plotten
    for i in range(len(L)):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Originale Magnetisierung mit Fehlerbalken
        ax.plot(beta, susz[i], "r-", label=f"L={L[i]}", linewidth=2)
        ax.fill_between(beta, susz[i] - susz_error[i], susz[i] + susz_error[i], color='r', alpha=0.2)

        # Renormierte Magnetisierung mit Fehlerbalken
        ax.plot(beta, all_suszre[18+i], "b-.", label=f"L'={L[i]}", linewidth=2)
        
        ax.fill_between(beta, all_suszre[4+i] , all_suszre[11+i] , color='b', alpha=0.2)

        # Achsenbeschriftungen und Titel
        ax.set_xlabel(r'$\beta$', fontsize=20)
        ax.set_ylabel(r'$|\chi|$', fontsize=20)
        ax.set_title(f'Reweighted Suszeptibility at $\\beta_c = {betaJ}$', fontsize=18, pad=20)

        # Legende
        ax.legend(fontsize=14)

        # Raster hinzufügen
        ax.grid(True, linestyle='--', alpha=0.5)

        # Achsenskalierung verbessern
        ax.set_xlim([beta.min(), beta.max()])
        #ax.set_ylim([min([m.min() for m in mag + magre]) - 0.02, max([m.max() for m in mag + magre]) + 0.02])

        # Layout verbessern
        plt.tight_layout()

        # Speichern der Abbildung
        #plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/test_inverse_complex_UNet/inverse_group_flow_UNet20000_susz_{i}.pdf', dpi=300, bbox_inches='tight')

        # Zeigen der Abbildung
        plt.show()
    """

    dsusz128 =  spip.interp1d(beta,susz128weigth)

    dsusz64re = spip.interp1d(beta,susz64re)
    dsusz32re = spip.interp1d(beta,susz32re)
    dsusz16re = spip.interp1d(beta,susz16re)

    beta_c = 0.5 * np.log(1 + np.sqrt(2)) 

    crit_renorm128_64 = np.log(dsusz64re(beta_c)/dsusz128(0.440436)) /(np.log(2)*(-1))
    crit_renorm128_32 = np.log(dsusz32re(beta_c)/dsusz128(0.44051)) /(np.log(2)*(-2))
    crit_renorm128_16 = np.log(dsusz16re(beta_c)/dsusz128(0.440538)) /(np.log(2)*(-3))
    crit_renorm64_32 = np.log(dsusz32re(beta_c)/dsusz64re(0.44051)) /(np.log(2)*(-1))
    crit_renorm32_16 = np.log(dsusz16re(beta_c)/dsusz32re(0.440538)) /(np.log(2)*(-1))
    crit_renorm64_16 = np.log(dsusz16re(beta_c)/dsusz64re(0.440538)) /(np.log(2)*(-2))

    crit_renorm128_64_err1 = -np.log(dmag64re(beta_c)/dmag128(0.43951)) /(np.log(2)*(-1))
    crit_renorm128_32_err1 = -np.log(dmag32re(beta_c)/dmag128(0.4399)) /(np.log(2)*(-2))
    crit_renorm128_16_err1 = -np.log(dmag16re(beta_c)/dmag128(0.44007)) /(np.log(2)*(-3))
    crit_renorm64_32_err1 = -np.log(dmag32re(beta_c)/dmag64re(0.4399)) /(np.log(2)*(-1))
    crit_renorm32_16_err1 = -np.log(dmag16re(beta_c)/dmag32re(0.44007)) /(np.log(2)*(-1))
    crit_renorm64_16_err1 = -np.log(dmag16re(beta_c)/dmag64re(0.44007)) /(np.log(2)*(-2))

    crit_renorm128_64_err2 = -np.log(dmag64re(beta_c)/dmag128(0.4418)) /(np.log(2)*(-1))
    crit_renorm128_32_err2 = -np.log(dmag32re(beta_c)/dmag128(0.44116)) /(np.log(2)*(-2))
    crit_renorm128_16_err2 = -np.log(dmag16re(beta_c)/dmag128(0.44105)) /(np.log(2)*(-3))
    crit_renorm64_32_err2 = -np.log(dmag32re(beta_c)/dmag64re(0.440116)) /(np.log(2)*(-1))
    crit_renorm32_16_err2 = -np.log(dmag16re(beta_c)/dmag32re(0.44105)) /(np.log(2)*(-1))
    crit_renorm64_16_err2 = -np.log(dmag16re(beta_c)/dmag64re(0.44105)) /(np.log(2)*(-2))

    print(crit_renorm128_64,crit_renorm128_32,crit_renorm128_16,crit_renorm64_32,crit_renorm64_16)

    data1 = np.array([crit_renorm128_64, crit_renorm128_32, crit_renorm128_16])
    data2 = np.array([crit_renorm128_64, crit_renorm64_32, crit_renorm32_16])
    data3 = np.array([crit_renorm32_16, crit_renorm64_16, crit_renorm128_16])

    err_up1 = data1 - np.array([crit_renorm128_64_err1, crit_renorm128_32_err1, crit_renorm128_16_err1])
    err_up2 = data2 - np.array([crit_renorm128_64_err1, crit_renorm64_32_err1, crit_renorm32_16_err1]) 
    err_up3 = data3 - np.array([crit_renorm32_16_err1, crit_renorm64_16_err1, crit_renorm128_16_err1]) 

    err_down1 = np.array([crit_renorm128_64_err2, crit_renorm128_32_err2, crit_renorm128_16_err2]) -data1
    err_down2 = np.array([crit_renorm128_64_err2, crit_renorm64_32_err2, crit_renorm32_16_err2]) -data2
    err_down3 = np.array([crit_renorm32_16_err2, crit_renorm64_16_err2, crit_renorm128_16_err2]) -data3

    # Calculate relative deviations
    relative_deviation1 = (data1 - 1.75) / 1.75
    relative_deviation2 = (data2 - 1.75) / 1.75
    relative_deviation3 = (data3 - 1.75) / 1.75

    relative_down1 = err_down1 / data1
    relative_down2 = err_down2 / data2
    relative_down3 = err_down3 / data3

    relative_dup1 = err_up1 / data1
    relative_dup2 = err_up2 / data2
    relative_dup3 = err_up3 / data3


    # Calculate errors
    errors1 = error_percent * relative_deviation1
    errors2 = error_percent * relative_deviation2
    errors3 = error_percent * relative_deviation3

    print( data1,err_up1,-err_down1)

    # Create figures with error bars
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].errorbar([1, 2, 3], relative_deviation1, yerr=[-relative_down1,relative_dup1],  fmt='o', capsize=8, elinewidth=2, label='Data Set 1')
    ax[0].set_title('Relative Deviation with Error Bars - Set 1', fontsize=16)
    ax[0].set_xlabel('Data Point', fontsize=14)
    ax[0].set_ylabel('Relative Deviation', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=12)

    ax[1].errorbar([1, 2, 3], relative_deviation2, yerr=[-relative_down2,relative_dup2], fmt='s', capsize=8, elinewidth=2, label='Data Set 2')
    ax[1].set_title('Relative Deviation with Error Bars - Set 2', fontsize=16)
    ax[1].set_xlabel('Data Point', fontsize=14)
    ax[1].set_ylabel('Relative Deviation', fontsize=14)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(fontsize=12)

    ax[2].errorbar([1, 2, 3], relative_deviation3, yerr=[-relative_down3,relative_dup3], fmt='^', capsize=8, elinewidth=2, label='Data Set 3')
    #ax[2].set_yscale('log')
    ax[2].set_title('Relative Deviation with Error Bars - Set 3', fontsize=16)
    ax[2].set_xlabel('Data Point', fontsize=14)
    ax[2].set_ylabel('Relative Deviation', fontsize=14)
    ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[2].legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/forward_crits_susz_rela.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Create figures with error bars
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].errorbar([1, 2, 3], data1, yerr=[-err_down1,err_up1],  fmt='o', capsize=8, elinewidth=2, label='Data Set 1')
    ax[0].axhline(y=1.75, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\gamma/\\nu$")
    ax[0].set_title('Relative Deviation with Error Bars - Set 1', fontsize=16)
    ax[0].set_xlabel('Data Point', fontsize=14)
    ax[0].set_ylabel('Relative Deviation', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=12)

    ax[1].errorbar([1, 2, 3], data2, yerr=[-err_down2,err_up2], fmt='s', capsize=8, elinewidth=2, label='Data Set 2')
    ax[1].axhline(y=1.75, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\gamma/\\nu$")
    ax[1].set_title('Relative Deviation with Error Bars - Set 2', fontsize=16)
    ax[1].set_xlabel('Data Point', fontsize=14)
    ax[1].set_ylabel('Relative Deviation', fontsize=14)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(fontsize=12)

    ax[2].errorbar([1, 2, 3], data3, yerr=[-err_down3,err_up3], fmt='^', capsize=8, elinewidth=2, label='Data Set 3')
    ax[2].axhline(y=1.75, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\gamma/\\nu$")
    ax[2].set_title('Relative Deviation with Error Bars - Set 3', fontsize=16)
    ax[2].set_xlabel('Data Point', fontsize=14)
    ax[2].set_ylabel('Relative Deviation', fontsize=14)
    ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[2].legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/forward_crits_susz_abs.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Einstellungen für einheitliche Schriftarten
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

    # Plotten
    for i in range(len(L)):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Originale Magnetisierung mit Fehlerbalken
        ax.plot(beta, susz[i], "r-", label=f"L={L[i]}", linewidth=2)
        ax.fill_between(beta, susz[i] - susz_error[i], susz[i] + susz_error[i], color='r', alpha=0.2)

        # Renormierte Magnetisierung mit Fehlerbalken
        ax.plot(beta, suszre[i], "b-.", label=f"L'={L[i]}", linewidth=2)
        ax.fill_between(beta, suszre[i] - suszre_error[i], suszre[i] + suszre_error[i], color='b', alpha=0.2)

        # Achsenbeschriftungen und Titel
        ax.set_xlabel(r'$\beta$', fontsize=16)
        ax.set_ylabel(r'$\chi$', fontsize=16)
        ax.set_title(f'Reweighted Suszeptibility at $\\beta_c = {betaJ}$', fontsize=18, pad=20)

        # Legende
        ax.legend(fontsize=14)

        # Raster hinzufügen
        ax.grid(True, linestyle='--', alpha=0.5)

        # Achsenskalierung verbessern
        ax.set_xlim([beta.min(), beta.max()])
        ax.set_ylim([min([m.min() for m in susz + suszre]) - 0.02, max([m.max() for m in susz + suszre]) + 0.02])

        # Layout verbessern
        plt.tight_layout()

        # Speichern der Abbildung
        plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/group_flow_{i}_susz.pdf', dpi=300, bbox_inches='tight')

        # Zeigen der Abbildung
        plt.show()

"""

def analyze_inverseRG(save = False):

    analysis()



if __name__ == '__main__':
    analyze_inverseRG(save = True)

    pass
