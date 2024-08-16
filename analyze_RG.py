
def analysis():

    length = 128
    betaJ = 0.44
    time_to_equilibrium = int(3e5)

    def mag_re(m_beta0,e_beta0,beta0,beta):
        m_re = np.empty_like(beta)
        for i in range(len(beta)):
            m_re[i] = np.sum(m_beta0 * np.exp(-(beta[i]-beta0) * e_beta0 ))  / np.sum(np.exp(-(beta[i] -beta0) *e_beta0 ))
                                                                                    
        return m_re
    


    L16_result = pickle.load(
        gzip.open(
        f'/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size16/betaJ{betaJ}/final_result/data_2e6.gz',
        mode = 'rb'
        )
    )

    energy16 = np.array((L16_result['energies']))[time_to_equilibrium:]
    mag16 = np.array((L16_result['magnetizations']))[time_to_equilibrium:]

    magnetization_mean_L16,magnetization_error_L16,time16 = \
        autocorr.calc_error(np.abs(mag16))

    L32_result = pickle.load(
        gzip.open(
        f'/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size32/betaJ{betaJ}/final_result/data_2e6.gz',
        mode = 'rb'
        )
    )

    energy32 = np.array((L32_result['energies']))[time_to_equilibrium:]
    mag32 = np.array((L32_result['magnetizations']))[time_to_equilibrium:]

    magnetization_mean_L32,magnetization_error_L32,time32 = \
        autocorr.calc_error(np.abs(mag32))

    L64_result = pickle.load(
        gzip.open(
        f'/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size64/betaJ{betaJ}/final_result/data_2e6.gz',
        mode = 'rb'
        )
    )

    energy64 = np.array((L64_result['energies']))[time_to_equilibrium:]
    mag64 = np.array((L64_result['magnetizations']))[time_to_equilibrium:]

    magnetization_mean_L64,magnetization_error_L64,time64 = \
        autocorr.calc_error(np.abs(mag64))
    


    L128_result = pickle.load(
        gzip.open(
        f'/tikhome/lspatscheck/Documents/bsc/simulation_data/lattice_size128/betaJ{betaJ}/final_result/data_2e6.gz',
        mode = 'rb'
        )
    )

    energy128 = np.array((L128_result['energies']))[time_to_equilibrium:]
    mag128 = np.array((L128_result['magnetizations']))[time_to_equilibrium:]



    energy128re = np.array((L128_result['energies']))[int(1.6e6)::2]
    mag128re = np.array((L128_result['magnetizations']))[int(1.6e6)::2]
    
    print(len(mag128))

    magnetization_mean_L128,magnetization_error_L128,time128 = \
        autocorr.calc_error(np.abs(mag128))

    config64= pickle.load(
    open(
         f'/data/lspatscheck/forward_renorm/128/config_renorm64.pickle',
         'rb'
        )
    )

    config32= pickle.load(
    open(
         f'/data/lspatscheck/forward_renorm/128/config_renorm32.pickle',
         'rb'
        )
    )

    config16= pickle.load(
    open(
         f'/data/lspatscheck/forward_renorm/128/config_renorm16.pickle',
         'rb'
        )
    )

    print("Data loaded") 

    magnetization_mean_L64re,magnetization_error_L64re,time64re = \
        autocorr.calc_error(np.abs(np.mean(config64,axis=(1,2))))
    
    magnetization_mean_L32re,magnetization_error_L32re,time32re = \
        autocorr.calc_error(np.abs(np.mean(config32,axis=(1,2))))

    magnetization_mean_L16re,magnetization_error_L16re,time16re = \
        autocorr.calc_error(np.abs(np.mean(config16,axis=(1,2))))

    beta = np.linspace(0.4395,0.4419,100)

    mag128weight = mag_re(np.abs(mag128),energy128,betaJ,beta)

    mag64weight = mag_re(np.abs(mag64),energy64,betaJ,beta)

    mag32weight = mag_re(np.abs(mag32),energy32,betaJ,beta)

    mag16weight = mag_re(np.abs(mag16),energy16,betaJ,beta)

    print("Done")
    mag64re = mag_re(np.abs(np.mean(config64,axis=(1,2))),energy128re,betaJ,beta)
    print("Done")
    mag32re = mag_re(np.abs(np.mean(config32,axis=(1,2))),energy128re,betaJ,beta)

    mag16re = mag_re(np.abs(np.mean(config16,axis=(1,2))),energy128re,betaJ,beta)

    mag = [mag64weight,mag32weight,mag16weight]
    magre = [mag64re,mag32re,mag16re]

    mag_error = [magnetization_error_L64,magnetization_error_L32,magnetization_error_L16]
    magre_error = [magnetization_error_L64re,magnetization_error_L32re,magnetization_error_L16re]

    dmag128 =  spip.interp1d(beta,mag128weight)

    dmag64re = spip.interp1d(beta,mag64re)
    dmag32re = spip.interp1d(beta,mag32re)
    dmag16re = spip.interp1d(beta,mag16re)

    beta_c = 0.5 * np.log(1 + np.sqrt(2)) 
    beta_c =0.44048

    crit_renorm128_64 = -np.log(dmag64re(beta_c )/dmag128(0.44048)) /(np.log(2)*(-1))
    crit_renorm128_32 = -np.log(dmag32re(beta_c )/dmag128(0.44048)) /(np.log(2)*(-2))
    crit_renorm128_16 = -np.log(dmag16re(beta_c )/dmag128(0.44048)) /(np.log(2)*(-3))
    crit_renorm64_32 = -np.log(dmag32re(beta_c )/dmag64re(0.44048)) /(np.log(2)*(-1))
    crit_renorm32_16 = -np.log(dmag16re(beta_c )/dmag32re(0.44048)) /(np.log(2)*(-1))
    crit_renorm64_16 = -np.log(dmag16re(beta_c )/dmag64re(0.44048)) /(np.log(2)*(-2))

    data = [crit_renorm32_16,crit_renorm64_16,crit_renorm128_16,crit_renorm32_16,crit_renorm64_32,crit_renorm128_64]    

    print(np.mean(np.array([crit_renorm128_64 ,crit_renorm128_32,crit_renorm128_16,crit_renorm64_32,crit_renorm32_16,crit_renorm64_16])),np.std(np.array([crit_renorm128_64 ,crit_renorm128_32,crit_renorm128_16,crit_renorm64_32,crit_renorm32_16,crit_renorm64_16])))

    pickle.dump(
        data,
        open(
            f'/tikhome/lspatscheck/Documents/bsc/simulation_data/standard_renorm/beta_L128.pickle',
            mode = 'wb'
        )
    )


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

    error_percent = 0.05

    print('Crits:',crit_renorm128_64 ,crit_renorm128_32,crit_renorm128_16,crit_renorm64_32,crit_renorm32_16,crit_renorm64_16)

    data1 = np.array([crit_renorm128_64, crit_renorm128_32, crit_renorm128_16])
    data2 = np.array([crit_renorm128_64, crit_renorm64_32, crit_renorm32_16])
    data3 = np.array([crit_renorm32_16, crit_renorm64_16, crit_renorm128_16])

    err_up1 = np.array([crit_renorm128_64_err1, crit_renorm128_32_err1, crit_renorm128_16_err1]) - data1
    err_up2 = np.array([crit_renorm128_64_err1, crit_renorm64_32_err1, crit_renorm32_16_err1]) - data2
    err_up3 = np.array([crit_renorm32_16_err1, crit_renorm64_16_err1, crit_renorm128_16_err1]) - data3

    err_down1 = np.array([crit_renorm128_64_err2, crit_renorm128_32_err2, crit_renorm128_16_err2]) -data1
    err_down2 = np.array([crit_renorm128_64_err2, crit_renorm64_32_err2, crit_renorm32_16_err2]) -data2
    err_down3 = np.array([crit_renorm32_16_err2, crit_renorm64_16_err2, crit_renorm128_16_err2]) -data3

    # Calculate relative deviations
    relative_deviation1 = (data1 - 0.125) / 0.125
    relative_deviation2 = (data2 - 0.125) / 0.125
    relative_deviation3 = (data3 - 0.125) / 0.125


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

    print(relative_deviation2,relative_down2,relative_dup2)

    # Create figures with error bars
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].errorbar([1, 2, 3], relative_deviation1, yerr=[-relative_down1,relative_dup1],  fmt='o', capsize=8, elinewidth=2, label='Data Set 1')
    ax[0].set_title('Relative Deviation with Error Bars - Set 1', fontsize=16)
    ax[0].set_xlabel('Data Point', fontsize=14)
    ax[0].set_ylabel('Relative Deviation', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=12)

    ax[1].errorbar([1, 2, 3], relative_deviation2, yerr=[np.array([-1,1,-1])*relative_down2,relative_dup2], fmt='s', capsize=8, elinewidth=2, label='Data Set 2')
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
    #plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/forward_crits_rela.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create figures with error bars
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].errorbar([1, 2, 3], data1, yerr=[-err_down1,err_up1],  fmt='o', capsize=8, elinewidth=2, label='Data Set 1')
    ax[0].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    ax[0].set_title('Relative Deviation with Error Bars - Set 1', fontsize=16)
    ax[0].set_xlabel('Data Point', fontsize=14)
    ax[0].set_ylabel('Relative Deviation', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=12)

    ax[1].errorbar([1, 2, 3], data2, yerr=[np.array([-1,1,-1])*err_down2,err_up2], fmt='s', capsize=8, elinewidth=2, label='Data Set 2')
    ax[1].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    ax[1].set_title('Relative Deviation with Error Bars - Set 2', fontsize=16)
    ax[1].set_xlabel('Data Point', fontsize=14)
    ax[1].set_ylabel('Relative Deviation', fontsize=14)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(fontsize=12)

    ax[2].errorbar([1, 2, 3], data3, yerr=[-err_down3,err_up3], fmt='^', capsize=8, elinewidth=2, label='Data Set 3')
    ax[2].axhline(y=0.125, color='gray', linestyle='--', linewidth=1,label = "Critical Exponent $\\beta/\\nu$")
    ax[2].set_title('Relative Deviation with Error Bars - Set 3', fontsize=16)
    ax[2].set_xlabel('Data Point', fontsize=14)
    ax[2].set_ylabel('Relative Deviation', fontsize=14)
    ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[2].legend(fontsize=12)

    plt.tight_layout()

    plt.show()

    L = [64,32,16]
# Einstellungen für einheitliche Schriftarten
    plt.rcParams.update({'font.size': 24, 'font.family': 'serif'})

    # Plotten
    for i in range(len(L)):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Originale Magnetisierung mit Fehlerbalken
        ax.plot(beta, mag[i], "r-", label=f"L={L[i]}", linewidth=2)
        ax.fill_between(beta, mag[i] - mag_error[i], mag[i] + mag_error[i], color='r', alpha=0.2)

        # Renormierte Magnetisierung mit Fehlerbalken
        ax.plot(beta, magre[i], "b-.", label=f"L'={L[i]}", linewidth=2)
        ax.fill_between(beta, magre[i] - magre_error[i], magre[i] + magre_error[i], color='b', alpha=0.2)

        # Achsenbeschriftungen und Titel
        ax.set_xlabel('$\\beta$', fontsize=24)
        ax.set_ylabel('$ \\langle \\left| m \\right| \\rangle $', fontsize=24)
        #ax.set_title(f'Reweighted Magnetization at $\\beta_c = {betaJ}$', fontsize=24, pad=20)

        # Legende
        ax.legend(fontsize=28)

        # Raster hinzufügen
        #ax.grid(True, linestyle='--', alpha=0.5)
        plt.xticks([0.4400,0.4406,0.4410], ["$0.4400$","$0.4406$","$0.4410$"], fontsize=24)
        # Achsenskalierung verbessern
        ax.set_xlim([0.440, 0.441])
        print([min([m.min() for m in mag[i] ]), max([m.max() for m in mag[i] ]) ])
        ax.set_ylim([min([m.min() for m in magre[i]-0.002 ]), max([m.max() for m in magre[i]+0.002 ]) ])

        # Layout verbessern
        plt.tight_layout()

        # Speichern der Abbildung
        plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/group_flow_{L[i]}.pdf', dpi=300, bbox_inches='tight')

        # Zeigen der Abbildung
        plt.show()
    

    mag_sq128weight = mag_re(np.abs(mag128*128**2)**2,energy128,betaJ,beta)

    mag_sq64weight = mag_re(np.abs(mag64*64**2)**2,energy64,betaJ,beta)

    mag_sq32weight = mag_re(np.abs(mag32*32**2)**2,energy32,betaJ,beta)

    mag_sq16weight = mag_re(np.abs(mag16*16**2)**2,energy16,betaJ,beta)


    mag_sq64re = mag_re(np.abs(np.mean(config64*64**2,axis=(1,2)))**2,energy128re,betaJ,beta)

    mag_sq32re = mag_re(np.abs(np.mean(config32*32**2,axis=(1,2)))**2,energy128re,betaJ,beta)

    mag_sq16re = mag_re(np.abs(np.mean(config16*16**2,axis=(1,2)))**2,energy128re,betaJ,beta)

    susz128weigth = (1/128**2) * ( mag_sq128weight - (mag128weight*128**2)**2   )

    susz64weigth = (1/64**2) * ( mag_sq64weight - (mag64weight*64**2)**2   )

    susz32weigth = (1/32**2) * ( mag_sq32weight - (mag32weight*32**2)**2   )

    susz16weigth = (1/16**2) * ( mag_sq16weight - (mag16weight*16**2)**2   )



    susz64re = (1/64**2) *( mag_sq64re - (mag64re*64**2)**2 )

    susz32re = (1/32**2) *(  mag_sq32re - (mag32re*32**2)**2  )

    susz16re = (1/16**2) *( mag_sq16re - (mag16re*16**2)**2    )

    susz_mean_L16re,susz_error_L16re,time16re = \
        autocorr.calc_error(susz16re)
    
    susz_mean_L32re,susz_error_L32re,time32re = \
        autocorr.calc_error(susz32re)
    
    susz_mean_L64re,susz_error_L64re,time64re = \
        autocorr.calc_error(susz16re)
    
    susz_mean_L16,susz_error_L16,time16 = \
        autocorr.calc_error(susz16weigth)
    
    susz_mean_L32,susz_error_L32,time32 = \
        autocorr.calc_error(susz32weigth)
    
    susz_mean_L64,susz_error_L64,time64 = \
        autocorr.calc_error(susz64weigth)
    



    susz = [susz64weigth,susz32weigth,susz16weigth]
    suszre = [susz64re,susz32re,susz16re]

    susz_error = [susz_error_L64,susz_error_L32,susz_error_L16]
    suszre_error = [susz_error_L64re,susz_error_L32re,susz_error_L16re]



    dsusz128 =  spip.interp1d(beta,susz128weigth)

    dsusz64re = spip.interp1d(beta,susz64re)
    dsusz32re = spip.interp1d(beta,susz32re)
    dsusz16re = spip.interp1d(beta,susz16re)

    beta_c =0.44048

    crit_renorm128_64 = np.log(dsusz64re(beta_c)/dsusz128(beta_c))/(np.log(2)*(-1))
    crit_renorm128_32 = np.log(dsusz32re(beta_c)/dsusz128(beta_c)) /(np.log(2)*(-2))
    crit_renorm128_16 = np.log(dsusz16re(beta_c)/dsusz128(beta_c)) /(np.log(2)*(-3))
    crit_renorm64_32 = np.log(dsusz32re(beta_c)/dsusz64re(beta_c)) /(np.log(2)*(-1))
    crit_renorm32_16 = np.log(dsusz16re(beta_c)/dsusz32re(beta_c)) /(np.log(2)*(-1))
    crit_renorm64_16 = np.log(dsusz16re(beta_c)/dsusz64re(beta_c)) /(np.log(2)*(-2))

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

    data = [crit_renorm32_16,crit_renorm64_16,crit_renorm128_16,crit_renorm32_16,crit_renorm64_32,crit_renorm128_64]    
    print(np.mean(np.array([crit_renorm128_64 ,crit_renorm128_32,crit_renorm128_16,crit_renorm64_32,crit_renorm32_16,crit_renorm64_16])),np.std(np.array([crit_renorm128_64 ,crit_renorm128_32,crit_renorm128_16,crit_renorm64_32,crit_renorm32_16,crit_renorm64_16])))
    print('Crits:',crit_renorm128_64 ,crit_renorm128_32,crit_renorm128_16,crit_renorm64_32,crit_renorm32_16,crit_renorm64_16)
    
    pickle.dump(
        data,
        open(
            f'/tikhome/lspatscheck/Documents/bsc/simulation_data/standard_renorm/gamma_L128t.pickle',
            mode = 'wb'
        )
    )

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
    #plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/forward_crits_susz_rela.png', dpi=300, bbox_inches='tight')
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
    #plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/forward_crits_susz_abs.png', dpi=300, bbox_inches='tight')
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
        plt.savefig(f'/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/group_flow_{L[i]}_susz.pdf', dpi=300, bbox_inches='tight')

        # Zeigen der Abbildung
        plt.show()


def analyze_RG(save = False):

    analysis()



if __name__ == '__main__':
    analyze_RG(save = True)

    pass
