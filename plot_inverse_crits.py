import numpy as np
import pickle
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as spip
import scipy.optimize as spopt
import autocorr


data= pickle.load(
open(
    f'/tikhome/lspatscheck/Documents/bsc/simulation_data/inverse_renorm/beta_crits.pickle',
    'rb'
    )
)

scheme_a = data["up16"]



print(scheme_a[1])

scheme_b = data["single_up"]



def progres_inc(series):
    means = np.empty(len(series))

    for i in range(len(series)):
        mean = (1/(i+1)) * np.sum(series[:i+1])
        means[i] = mean
    return means

import matplotlib.pyplot as plt

# Define the x-axis values
x_values = np.array([32, 64, 128, 256, 512, 1024, 2048])

label_added_x = False
label_added_o = False

plt.rcParams.update({'font.size': 24, 'font.family': 'serif'})

plt.figure(figsize=(10, 9))

for i in range(len(scheme_a)):
    y_values = np.abs((progres_inc(scheme_a[i]) - 0.125) / 0.125)

    print(y_values)
    if i in [0, 4, 6, 7,8]:

        if not label_added_x:
            plt.plot(x_values, y_values, 'rx--', label=' "not converging" ',markeredgewidth=4, markersize=12)
            label_added_x = True
        else:
            plt.plot(x_values, y_values, 'rx--',markeredgewidth=4, markersize=12)
    else:
        if not label_added_o:
            plt.plot(x_values, y_values, 'o--',color = 'blue', markerfacecolor='none',  markeredgewidth=4, markersize=12,label=' "converging" ')
            label_added_o = True
        else:
            plt.plot(x_values, y_values,'o--', color = 'blue', markerfacecolor='none', markeredgewidth=4, markersize=12)

#plt.loglog()
#plt.semilogx()
#plt.yscale('symlog', linthresh=0.006)
plt.loglog()

plt.xlabel('$L_\\mathrm{max}$', fontsize=24)
plt.ylabel('$\\Delta_\\mathrm{rela} \\left( \\beta_m / \\nu \\right)$', fontsize=24)
#plt.title('Relative Difference Over Maximal Included System Size', fontsize=22)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)

#plt.yticks([0,0.1], ["$0$","$0.01$"], fontsize=24)
plt.legend(loc='best', fontsize=24)
plt.tight_layout()

# Save the plot
plt.savefig('/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/scheme_a_plot_log.pdf', dpi=300)

plt.show()

label_added_x = False
label_added_o = False

plt.rcParams.update({'font.size': 24, 'font.family': 'serif'})
# Plot for scheme_b
plt.figure(figsize=(10, 9))

for i in range(len(scheme_b)):
    y_values = np.abs((progres_inc(scheme_b[i]) - 0.125) / 0.125)

    if i in [0, 4, 6, 7,8]:
        if not label_added_x:
            plt.plot(x_values, y_values, 'rx--', label=' "not converging" ',markeredgewidth=4, markersize=12)
            label_added_x = True
        else:
            plt.plot(x_values, y_values, 'rx--',markeredgewidth=4, markersize=12)
    else:
        print(y_values)
        if not label_added_o:
            plt.plot(x_values, y_values, 'o--',color = 'green', markerfacecolor='none',markeredgewidth=4, markersize=12,label=' "converging" ')
            label_added_o = True
        else:
            plt.plot(x_values, y_values,'o--', color = 'green', markerfacecolor='none',markeredgewidth=4, markersize=12)


plt.loglog()
plt.xlabel('$L_\\mathrm{max}$', fontsize=24)
plt.ylabel('$\\Delta_\\mathrm{rela} \\left( \\beta_m / \\nu \\right)$', fontsize=24)
#plt.title('Relative Difference Over Maximal Included System Size', fontsize=22)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best', fontsize=24)
plt.tight_layout()

# Save the plot
plt.savefig('/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/scheme_b_plot_log.pdf', dpi=300)

plt.show()


stacked_scheme_a = np.vstack((progres_inc(scheme_a[0]),progres_inc(scheme_a[1]),progres_inc(scheme_a[2]),progres_inc(scheme_a[3]),progres_inc(scheme_a[4]),progres_inc(scheme_a[5]),progres_inc(scheme_a[6]),progres_inc(scheme_a[7]),progres_inc(scheme_a[8])))
# Berechnen des Mittelwerts zu jedem Zeitpunkt (entlang der ersten Achse)
mean_scheme_a = np.mean(stacked_scheme_a, axis=0)
err_scheme_a = np.std(stacked_scheme_a,axis= 0)/ np.sqrt(9)

stacked_scheme_b = np.vstack((progres_inc(scheme_b[0]),progres_inc(scheme_b[1]),progres_inc(scheme_b[2]),progres_inc(scheme_b[3]),progres_inc(scheme_b[4]),progres_inc(scheme_b[5]),progres_inc(scheme_b[6]),progres_inc(scheme_b[7]),progres_inc(scheme_b[8])))
# Berechnen des Mittelwerts zu jedem Zeitpunkt (entlang der ersten Achse)
mean_scheme_b = np.mean(stacked_scheme_b, axis=0)
err_scheme_b = np.std(stacked_scheme_b,axis= 0)/ np.sqrt(9)



plt.rcParams.update({'font.size': 24, 'font.family': 'serif'})
plt.figure(figsize=(10, 9))



plt.errorbar(x_values-0.1, (mean_scheme_a - 0.125) / 0.125, yerr=err_scheme_a / 0.125, 
             fmt='o', color = 'blue' , capsize=5, elinewidth=2, markeredgewidth=2, markersize=12, label='Scheme A')

plt.errorbar(x_values+0.1, (mean_scheme_b - 0.125) / 0.125, yerr=err_scheme_b / 0.125, 
             fmt='o', color = 'green', capsize=5, elinewidth=2, markeredgewidth=2, markersize=12, label='Scheme B')



plt.axhline(y=0.01626, color='r', linestyle='--', linewidth=1, label='Standard RG')
plt.fill_between( x_values, np.ones(len(x_values)) *(0.01626 - 0.0018), np.ones(len(x_values)) *(0.01626 + 0.0018), color='r', alpha=0.2)


#plt.semilogx()
plt.loglog()
plt.xlabel('$L_\\mathrm{max}$', fontsize=24)
plt.ylabel('$ \\langle \\Delta_\\mathrm{rela} \\left( \\beta_m / \\nu \\right) \\rangle$', fontsize=24)
#plt.title('Relative Difference Over Maximal Included System Size', fontsize=22)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best', fontsize=24)
plt.tight_layout()

# Save the plot
plt.savefig('/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/scheme_mean_all.pdf', dpi=300)

stacked_scheme_a = np.vstack((progres_inc(scheme_a[1]),progres_inc(scheme_a[2]),progres_inc(scheme_a[3]),progres_inc(scheme_a[5])))
# Berechnen des Mittelwerts zu jedem Zeitpunkt (entlang der ersten Achse)
mean_scheme_a = np.mean(stacked_scheme_a, axis=0)
err_scheme_a = np.std(stacked_scheme_a,axis= 0)/ np.sqrt(4*np.array([1,2,3,4,5,6,7]))

stacked_scheme_b = np.vstack((progres_inc(scheme_b[1]),progres_inc(scheme_b[2]),progres_inc(scheme_b[3]),progres_inc(scheme_a[5])))
mean_scheme_b = np.mean(stacked_scheme_b, axis=0)
err_scheme_b = np.std(stacked_scheme_b,axis= 0)/ np.sqrt(4*np.array([1,2,3,4,5,6,7]))

print(mean_scheme_a,mean_scheme_b)
print(err_scheme_a,err_scheme_b)

plt.rcParams.update({'font.size': 24, 'font.family': 'serif'})
plt.figure(figsize=(10, 9))



plt.errorbar(x_values-0.1, (mean_scheme_a - 0.125) / 0.125, yerr=err_scheme_a / 0.125, 
             fmt='o', color = 'blue' , capsize=5, elinewidth=2, markeredgewidth=2, markersize=12, label='Scheme A')
print("Results:",mean_scheme_b )
plt.errorbar(x_values+0.1, (mean_scheme_b - 0.125) / 0.125, yerr=err_scheme_b / 0.125, 
             fmt='o', color = 'green', capsize=5, elinewidth=2, markeredgewidth=2, markersize=12, label='Scheme B')


print("Gamma:")

print((mean_scheme_a , err_scheme_a))
print((mean_scheme_a - 0.125) / 0.125, err_scheme_a / 0.125)

print((mean_scheme_b , err_scheme_b))
print((mean_scheme_b - 0.125) / 0.125, err_scheme_b / 0.125)


plt.axhline(y=0.00973, color='r', linestyle='--', linewidth=1, label='Standard RG')

plt.axhline(y=0.00, linestyle='--', linewidth=1, label='Analytic')

x_min, x_max = plt.xlim(x_values.min()-10, x_values.max()+1000)

# Create a range covering the entire x-axis
x_full_range = np.linspace(x_min, x_max, 1000)

# Extend the fill_between over the entire x-axis range


plt.fill_between( x_full_range, 0.00973 - 0.00675, 0.00973 + 0.00675, color='r', alpha=0.2)


#plt.loglog()
plt.semilogx()

plt.xlabel('$L_\\mathrm{max}$', fontsize=24)
plt.ylabel('$ \\langle \\Delta_\\mathrm{rela} \\left( \\beta_m / \\nu \\right) \\rangle$', fontsize=24)
#plt.title('Relative Difference Over Maximal Included System Size', fontsize=22)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best', fontsize=24)
plt.tight_layout()

# Save the plot
plt.savefig('/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/scheme_mean.pdf', dpi=300)

plt.show()


#### loglog of mean schema_a/b filteres


plt.rcParams.update({'font.size': 24, 'font.family': 'serif'})
plt.figure(figsize=(10, 9))



plt.errorbar(x_values-0.1, (mean_scheme_a - 0.125) / 0.125, yerr=err_scheme_a / 0.125, 
             fmt='o', color = 'blue' , capsize=5, elinewidth=2, markeredgewidth=2, markersize=12, label='Scheme A')
print("Results:",mean_scheme_b )
plt.errorbar(x_values+0.1, (mean_scheme_b - 0.125) / 0.125, yerr=err_scheme_b / 0.125, 
             fmt='o', color = 'green', capsize=5, elinewidth=2, markeredgewidth=2, markersize=12, label='Scheme B')



plt.axhline(y=0.00973, color='r', linestyle='--', linewidth=1, label='Standard RG')



x_min, x_max = plt.xlim(x_values.min()-10, x_values.max()+1000)

# Create a range covering the entire x-axis
x_full_range = np.linspace(x_min, x_max, 1000)

# Extend the fill_between over the entire x-axis range


plt.fill_between( x_full_range, 0.00973 - 0.00675, 0.00973 + 0.00675, color='r', alpha=0.2)


plt.loglog()

plt.xlabel('$L_\\mathrm{max}$', fontsize=24)
plt.ylabel('$ \\langle \\Delta_\\mathrm{rela} \\left( \\beta_m / \\nu \\right) \\rangle$', fontsize=24)
#plt.title('Relative Difference Over Maximal Included System Size', fontsize=22)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best', fontsize=24)
plt.tight_layout()

# Save the plot
plt.savefig('/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/scheme_mean_loglog.pdf', dpi=300)

plt.show()


########################################################

plt.rcParams.update({'font.size': 24, 'font.family': 'serif'})

plt.figure(figsize=(10, 6))



plt.errorbar(x_values, (mean_scheme_a - 0.125) / 0.125, yerr=err_scheme_a / 0.125, 
             fmt='o', capsize=5, elinewidth=2, markeredgewidth=2, label='Scheme B')

plt.semilogx()

plt.xlabel('System Size', fontsize=14)
plt.ylabel('Relative Deviation from 0.125', fontsize=24)
plt.title('Relative Deviation of Scheme B', fontsize=24)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best', fontsize=24)
plt.tight_layout()

# Save the plot
plt.savefig('/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/scheme_b_mean.pdf', dpi=300)

plt.show()



data= pickle.load(
open(
    f'/tikhome/lspatscheck/Documents/bsc/simulation_data/inverse_renorm/gamma_crits.pickle',
    'rb'
    )
)

scheme_a = data["up16"]

print(scheme_a[1])

scheme_b = data["single_up"]

def progres_inc(series):
    means = np.empty(len(series))

    for i in range(len(series)):
        mean = (1/(i+1)) * np.sum(series[:i+1])
        means[i] = mean
    return means

import matplotlib.pyplot as plt

label_added_x = False
label_added_o = False
# Plot for scheme_a

plt.rcParams.update({'font.size': 24, 'font.family': 'serif'})

plt.figure(figsize=(10, 9))

for i in range(len(scheme_a)):
    y_values = np.abs((progres_inc(scheme_a[i]) - 1.75) / 1.75)
    if i in [0, 4, 6, 7,8]:
        if not label_added_x:
            plt.plot(x_values, y_values, 'rx--', label=' "not converging" ',markeredgewidth=4, markersize=12)
            label_added_x = True
        else:
            plt.plot(x_values, y_values, 'rx--',markeredgewidth=4, markersize=12)
    else:
        if not label_added_o:
            plt.plot(x_values, y_values, 'o--',color = 'blue', markerfacecolor='none', markeredgewidth=4, markersize=12,label=' "converging "')
            label_added_o = True
        else:
            plt.plot(x_values, y_values,'o--', color = 'blue', markerfacecolor='none',markeredgewidth=4, markersize=12)

#plt.semilogx()
plt.loglog()
plt.xlabel('$L_\\mathrm{max}$', fontsize=24)
plt.ylabel('$\\Delta_\\mathrm{rela} \\left( \\gamma / \\nu \\right) $', fontsize=24)
#plt.title('Relative Deviation of Scheme A', fontsize=24)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best', fontsize=24)
plt.tight_layout()

# Save the plot
plt.savefig('/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/scheme_a_plot_gamma.pdf', dpi=300)

plt.show()


label_added_x = False
label_added_o = False

# Plot for scheme_b
plt.rcParams.update({'font.size': 24, 'font.family': 'serif'})

plt.figure(figsize=(10,9))

for i in range(len(scheme_b)):
    y_values = np.abs((progres_inc(scheme_b[i]) - 1.75) / 1.75)
    if i in [0, 4, 6, 7,8]:
        if not label_added_x:
            plt.plot(x_values, y_values, 'rx--', label=' "not converging" ',markeredgewidth=4, markersize=12)
            label_added_x = True
        else:
            plt.plot(x_values, y_values, 'rx--',markeredgewidth=4, markersize=12)
    else:
        if not label_added_o:
            plt.plot(x_values, y_values, 'o--',color = 'green', markerfacecolor='none',markeredgewidth=4, markersize=12,label=' "converging" ')
            label_added_o = True
        else:
            plt.plot(x_values, y_values,'o--', color = 'green', markerfacecolor='none',markeredgewidth=4, markersize=12)

#plt.semilogx()
plt.loglog()
plt.xlabel('$L_\\mathrm{max}$', fontsize=24)
plt.ylabel('$\\Delta_\\mathrm{rela} \\left( \\gamma / \\nu \\right)$', fontsize=24)
#plt.title('Relative Deviation of Scheme B', fontsize=24)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best', fontsize=24)
plt.tight_layout()

# Save the plot
plt.savefig('/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/scheme_b_plot_gamma.pdf', dpi=300)

plt.show()

stacked_scheme_a = np.vstack((progres_inc(scheme_a[1]),progres_inc(scheme_a[2]),progres_inc(scheme_a[3]),progres_inc(scheme_a[5])))
# Berechnen des Mittelwerts zu jedem Zeitpunkt (entlang der ersten Achse)
mean_scheme_a = np.mean(stacked_scheme_a, axis=0)
err_scheme_a = np.std(stacked_scheme_a,axis= 0)/ np.sqrt(4*np.array([1,2,3,4,5,6,7]))

stacked_scheme_b = np.vstack((progres_inc(scheme_b[1]),progres_inc(scheme_b[2]),progres_inc(scheme_b[3]),progres_inc(scheme_b[5])))
# Berechnen des Mittelwerts zu jedem Zeitpunkt (entlang der ersten Achse)
mean_scheme_b = np.mean(stacked_scheme_b, axis=0)
err_scheme_b = np.std(stacked_scheme_b,axis= 0)/ np.sqrt(4*np.array([1,2,3,4,5,6,7]))

print(mean_scheme_a,mean_scheme_b)
print(err_scheme_a,err_scheme_b)


plt.rcParams.update({'font.size': 24, 'font.family': 'serif'})
plt.figure(figsize=(10, 9))






plt.errorbar(x_values-0.1, (mean_scheme_a - 1.75) / 1.75, yerr=err_scheme_a / 1.75, 
             fmt='bo', capsize=3, elinewidth=1,markeredgewidth=2, markersize=12, label='Scheme A')

plt.errorbar(x_values+0.1, (mean_scheme_b - 1.75) / 1.75, yerr=err_scheme_b / 1.75, 
             fmt='go', capsize=3, elinewidth=1,markeredgewidth=2, markersize=12,label='Scheme B')

print("Gamma:")

print((mean_scheme_a , err_scheme_a))
print((mean_scheme_a - 1.75) / 1.75, err_scheme_a / 1.75)

print((mean_scheme_b , err_scheme_b))
print((mean_scheme_b - 1.75) / 1.75, err_scheme_b / 1.75)

plt.axhline(y=0.0115, color='r', linestyle='--', linewidth=1, label='Standard RG')


plt.axhline(y=0.0, linestyle='--', linewidth=1, label='Analytic')

# Define the full x-axis range explicitly
x_min, x_max = plt.xlim(x_values.min()-10, x_values.max()+1000)

# Create a range covering the entire x-axis
x_full_range = np.linspace(x_min, x_max, 1000)

# Extend the fill_between over the entire x-axis range
plt.fill_between(x_full_range, 0.0109 - 0.00632, 0.0109 + 0.00632, color='r', alpha=0.2)


plt.semilogx()
#plt.loglog()
plt.xlabel('$L_\\mathrm{max}$', fontsize=24)
plt.ylabel('$ \\langle \\Delta_\\mathrm{rela} \\left( \\gamma / \\nu \\right) \\rangle$', fontsize=24)
#plt.title('Relative Deviation of Scheme A', fontsize=24)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best', fontsize=24)
plt.tight_layout()
plt.savefig('/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/scheme_mean_gamma.pdf', dpi=300)
plt.show()

###########################

plt.rcParams.update({'font.size': 24, 'font.family': 'serif'})
plt.figure(figsize=(10, 9))






plt.errorbar(x_values-0.1, (mean_scheme_a - 1.75) / 1.75, yerr=err_scheme_a / 1.75, 
             fmt='bo', capsize=3, elinewidth=1,markeredgewidth=2, markersize=12, label='Scheme A')

plt.errorbar(x_values+0.1, (mean_scheme_b - 1.75) / 1.75, yerr=err_scheme_b / 1.75, 
             fmt='go', capsize=3, elinewidth=1,markeredgewidth=2, markersize=12,label='Scheme B')

plt.axhline(y=0.0115, color='r', linestyle='--', linewidth=1, label='Standard RG')


# Define the full x-axis range explicitly
x_min, x_max = plt.xlim(x_values.min()-10, x_values.max()+1000)

# Create a range covering the entire x-axis
x_full_range = np.linspace(x_min, x_max, 1000)

# Extend the fill_between over the entire x-axis range
plt.fill_between(x_full_range, 0.0109 - 0.00632, 0.0109 + 0.00632, color='r', alpha=0.2)


#plt.semilogx()
plt.loglog()
plt.xlabel('$L_\\mathrm{max}$', fontsize=24)
plt.ylabel('$ \\langle \\Delta_\\mathrm{rela} \\left( \\gamma / \\nu \\right) \\rangle$', fontsize=24)
#plt.title('Relative Deviation of Scheme A', fontsize=24)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best', fontsize=24)
plt.tight_layout()
plt.savefig('/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/scheme_mean_gamma_loglog.pdf', dpi=300)
plt.show()

##########################

plt.figure(figsize=(10, 6))



plt.plot(x_values, mean_scheme_b, 'o', label=f'Scheme A - Series {i}')

#plt.semilogx()
plt.loglog()

plt.xlabel('System Size', fontsize=14)
plt.ylabel('Relative Deviation from 1.75', fontsize=14)
plt.title('Relative Deviation of Scheme B', fontsize=16)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()

# Save the plot
#plt.savefig('/tikhome/lspatscheck/Documents/bsc/plots_for_presentation/scheme_a_plot.pdf', dpi=300)

plt.show()
