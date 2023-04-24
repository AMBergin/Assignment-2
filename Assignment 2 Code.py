import numpy as np
import matplotlib.pyplot as plt

times = 1000
def metropolis(beta, n, x_0=0):
    x = x_0 
    H = x**2
    HList = [H]
    for i in range(n):
        displacement = np.random.normal(0, 1)
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

def energy(beta, n, N): #n is the number of steps and N is the number of samples taken
    Hs = np.zeros(N) 
    for i in range(N):
        HList = metropolis(beta, n)
        Hs[i] = np.mean(HList)
    return np.mean(Hs), np.std(Hs) #this collects the mean of the results ensuring it is as accurate as possible

def reweight(Beta1, Beta2, E1, P1):
    P2_unweighted = np.sum(np.exp(-Beta2 * metropolis(Beta2, times)))
    P2 = np.sum(np.exp(-Beta1 * metropolis(Beta2, times) - (Beta2 - Beta1) * E1))
    E2 = np.sum(metropolis(Beta2, times) * np.exp(-Beta2 * metropolis(Beta2, times))) / P2_unweighted
    return E1 * np.exp((Beta1 - Beta2) * E1) / P1 * np.exp((Beta2 - Beta1) * E2) / P2
 #this is the reweighting procedure using the formula provided

BetaList= np.linspace(1, 10, 100)
Hs= np.zeros(len(BetaList))
HReweight = np.zeros(len(BetaList))
DErrors = np.zeros(len(BetaList))
RErrors = np.zeros(len(BetaList))
for i, beta in enumerate(BetaList):
    Hs[i], DErrors[i]= energy(beta, times, 10) #this compiles all results into a single list for plotting
    if i == 0: #this compiles the reweighted data for plotting
        P1= np.sum(np.exp(-BetaList[i]*metropolis(BetaList[i], times)))
        HReweight[i]=Hs[i] 
        RErrors[i] = DErrors[i]
    else:
        HReweight[i]=reweight(BetaList[i-1], BetaList[i],HReweight[i-1], P1)
        RErrors[i] = DErrors[i-1] * np.sqrt(np.exp(-2 * (BetaList[i] - BetaList[i-1]) * HReweight[i-1] / P1))

plt.errorbar(BetaList, Hs, yerr=DErrors, fmt='o', label='Direct')
plt.errorbar(BetaList, HReweight, yerr=RErrors, fmt='o', label='Reweight')
plt.xlabel('Inverse Temperature (Beta)')
plt.ylabel('Average Energy (E)')
plt.title('Gaussian System with Hamiltonian H=x^2')
plt.legend()
plt.show()
