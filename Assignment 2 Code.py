
import numpy as np
import matplotlib.pyplot as plt

times = 1000
def metropolis(beta, n, x_0=0):
    x = x_0 
    H = x**2
    HList = [H]
    for i in range(n):
        displacement = np.random.normal(-1, 1)
        x_1 = x+displacement #candidate
        H_1 = x_1**2 #candidate energy
        DeltaH = H_1 - H 
        if DeltaH <= 0:
            A = 1 #acceptance probability
        else:
            A= np.exp(-beta * DeltaH)
        if np.random.uniform(0,1) < A:
            x = x_1 #if accepted then the starting value changes
            H = H_1 #as does the energy
        HList.append(H) #if not accepted then the value is not added to the list and the previous value is accepted 
        # at the beginning of this function
    return np.array(HList)

def energy(beta1, beta2, n, N):
    Hs = np.zeros(N)
    HReweight = np.zeros(N)
    for i in range(N):
        HList = metropolis(beta1, n)
        Hs[i] = np.mean(HList)
        weights = np.exp(-(beta2 - beta1) * HList)
        HReweight[i] = np.mean(HList * weights) / np.mean(weights)
    return np.mean(Hs), np.std(Hs), np.mean(HReweight), np.std(HReweight) #this collects the mean of the results ensuring it is as accurate as possible

Beta1List = np.linspace(1, 10, 100)
Hs = np.zeros(len(Beta1List))
HReweight = np.zeros(len(Beta1List))
DErrors = np.zeros(len(Beta1List))
RErrors = np.zeros(len(Beta1List))
for i, beta1 in enumerate(Beta1List):
    Hs[i], DErrors[i], HReweight[i], RErrors[i] = energy(beta1, 1+i/10, times, 10)
 #this compiles all results into a single list for plotting
    

plt.errorbar(Beta1List, Hs, yerr=DErrors, fmt='o', label='Direct')
plt.errorbar(Beta1List, HReweight, yerr=RErrors, fmt='o', label='Reweight')
plt.xlabel('Inverse Temperature (Beta)')
plt.ylabel('Average Energy (E)')
plt.title('Gaussian System with Hamiltonian H=x^2')
plt.legend()
plt.show()
