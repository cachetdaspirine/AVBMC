#!/usr/bin/python3
#!/home/hugo/anaconda3/bin/python3

from System import *
from McMove import *
import time
import os
import sys

if len(sys.argv)<2:
    print('Number of the serie not specified, please enter a serie name')
    sys.exit()

time_start = time.perf_counter()
SimNum=sys.argv[1]
sys.path.insert(0,'Res/Sim'+str(SimNum))
from Parameter import *
#os.system('rm -rf Res/Sim'+str(SimNum))
#os.system('mkdir Res/Sim'+str(SimNum))

Output=False
with open('Res/Sim'+str(SimNum)+'/Energy.out','w') as myfile:
    myfile.write('time ElasticEnergy SurfaceEnergy TotalEnergy \n')



def CoolDown(time,DE0):
    #print(DE0)
    if DE0!=0:
        return -6/(7*DE0)*np.log(1-time/TimeStepTot)+1/(7*DE0)
    else :
        return BetaInitial
    #return BetaInitial+time/TimeStepTot*(BetaFinal-BetaInitial)



#   ___    _   _   _____   ____    _   _   _____
#  / _ \  | | | | |_   _| |  _ \  | | | | |_   _|
# | | | | | | | |   | |   | |_) | | | | |   | |
# | |_| | | |_| |   | |   |  __/  | |_| |   | |
#  \___/   \___/    |_|   |_|      \___/    |_|

with open('Res/Sim'+str(SimNum)+'/Parameter.out','w') as myfile:
    myfile.write('TimeStepTot '+str(TimeStepTot)+'\n')
    myfile.write('StatTime '+str(StatTime)+'\n')
    myfile.write('BetaInitial '+str(BetaInitial)+'\n')
    myfile.write('BetaFinal '+str(BetaFinal)+'\n')
    myfile.write('DE0G '+str(DEG)+'\n')
    myfile.write('Kmain '+str(Kmain)+'\n')
    myfile.write('Kcoupling '+str(Kcoupling)+'\n')
    myfile.write('Eps '+str(Eps)+'\n')
    myfile.write('KVOL '+str(KVOL)+'\n')
    myfile.write('J '+str(J)+'\n')
    myfile.write('SizeX '+str(SizeX)+'\n')
    myfile.write('SizeY '+str(SizeY)+'\n')
    myfile.write('NumberOfParticles '+str(NumberOfParticle)+'\n')

#  ___           _   _     _           _   _                  ____                  _
# |_ _|  _ __   (_) | |_  (_)   __ _  | | (_)  ____   ___    / ___|   _   _   ___  | |_    ___   _ __ ___
#  | |  | '_ \  | | | __| | |  / _` | | | | | |_  /  / _ \   \___ \  | | | | / __| | __|  / _ \ | '_ ` _ \
#  | |  | | | | | | | |_  | | | (_| | | | | |  / /  |  __/    ___) | | |_| | \__ \ | |_  |  __/ | | | | | |
# |___| |_| |_| |_|  \__| |_|  \__,_| |_| |_| /___|  \___|   |____/   \__, | |___/  \__|  \___| |_| |_| |_|
#                                                                    |___/
rd.seed(Seed)
np.random.seed(Seed)
Beta=BetaInitial
Syst=System(SizeX,SizeY,J=J,Eps=Eps,Kcoupling=Kcoupling,Kmain=Kmain,Kvol=KVOL)
MC=MonteCarlo(NumberOfParticle,SimNum)
for n in range(NumberOfParticle):
    Syst.AddRandParticle()

print(" __  __           _             _                             ")
print("|  \/  |   __ _  (_)  _ __     | |       ___     ___    _ __  ")
print("| |\/| |  / _` | | | | '_ \    | |      / _ \   / _ \  | '_ \ ")
print("| |  | | | (_| | | | | | | |   | |___  | (_) | | (_) | | |_) |")
print("|_|  |_|  \__,_| |_| |_| |_|   |_____|  \___/   \___/  | .__/ ")
print("                                                       |_|    ")
Beta=0
for t in range(1,TimeStepTot):
    Success=True
    #------Energy before the move---------------------
    Ei=Syst.Compute_Energy()
    #------Make the move------------------------------
    MC.McMove(Syst)
    #------Store the Energy after the move------------
    Eaft=Syst.Compute_Energy()
    #------see wether we accept the move or not-------
    if rd.uniform(0,1)>np.exp(-(Eaft-Ei)*Beta) :
        #--Move refused-------------------------------
        Syst=MC.Reverse()
        Success=False
    #------keep track of the success/fail-------------
    MC.Count(Success,Eaft-Ei)
    #------Cool down the system ----------------------
    if t>StatTime:
        Beta=CoolDown(t,MC.avDE/5.)
    #------Make the stats and adapt the McMove--------
    if t%StatTime==0 and t!=0 :
        print("time=",t)
        MC.MakeStat(t,Beta)
        if Syst.g_Np()!=0:
            with open('Res/Sim'+str(SimNum)+'/Energy.out','a') as myfile:
                myfile.write(str(t)+" "+str(Syst.ElasticEnergy/Syst.g_Np())+" "+str(Syst.SurfaceEnergy/Syst.g_Np())+" "+str((Syst.ElasticEnergy+Syst.SurfaceEnergy)/Syst.g_Np())+"\n")
            if Output:
                Syst.PrintPerSite('Res/Sim'+str(SimNum)+'/Site_time'+str(t)+'.res',Path='Res/Sim'+str(SimNum)+'/')
            #Syst.PrintPerSpring('Res/Sim'+str(SimNum)+'/Spring_time'+str(t)+'.res')
#Syst.PlotPerSite()
Syst.PrintPerSite('Res/Sim'+str(SimNum)+'/Site_Final.res')
#Syst.PrintPerSpring('Res/Sim'+str(SimNum)+'/Spring_Final.res')
#Syst.PlotPerSite()

print(time.perf_counter() - time_start)
