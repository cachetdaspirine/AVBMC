import numpy as np
import random as rd
from System import *
import os

class MonteCarlo:
    def __init__(self,Np=1,SimNum=0,Pbias=0):
        self.Pbias=Pbias
        self.Success=0
        self.Refuse=0
        self.DEP,self.DEN=0,0
        self.DEPA=0 #number of step with DE positiv and accepted
        self.DE=0
        self.radius=np.inf
        self.Nmove=min(Np,10)
        self.Np=Np
        self.SimNum=SimNum
        self.CopySystem=System()
        with open('Res/Sim'+str(self.SimNum)+'/Stat.out','w') as myfile:
            myfile.write('time Beta AcceptanceRate RefusalRate Nmove Radius\n')
        with open('Res/Sim'+str(self.SimNum)+'/AdvanceStat.out','w') as myfile:
            myfile.write('time Beta PositiveDERate NegativeDERate AcceptedPositiveDERate averageDE\n')
    def McMove(self,BinSyst):
        self.CopySystem=System(Old_System=BinSyst)
        for _ in range(self.Nmove):
            IJ0 = BinSyst.RemoveRandParticle()
            BinSyst.AddRandParticle(IJ0,self.radius)
    def McClusterMove(self,BinSyst): # works but slow
        self.CopySystem=System(Old_System=BinSyst)
        for _ in range(self.Nmove):
            # This garanti that removing a particle wont split the cluster in two parts
            # So only BinSyst.BinaryClusters[-1] will be affected
            AffectedCluster=BinSyst.RmRandContiguousParticle()
            BinSyst.AddParticleToCluster(AffectedCluster)
                #continue
            #BinSyst.CheckClusterToSite()
    def McMoveInOut(self,BinSyst):
        self.CopySystem=System(Old_System=BinSyst)
        self.Prob=1
        for _ in range(self.Nmove):
            NIJ = BinSyst.SelectRandomNeighbor()
            InBefore= BinSyst.RemoveRandParticle(NIJ=NIJ)
            if rd.uniform(0,1)<self.Pbias:
                #Add a particle in the vicinity of NIJ
                BinSyst.AddParticleVicinity(NIJ)
                InAfter=True
            else:
                #Add a particle out of the vicinity of NIJ
                try:
                    BinSyst.AddParticleOutVicinity(NIJ)
                    InAfter=False
                except ValueError:
                    print('Add Random particle')
                    BinSyst.AddRandParticle()
                    InAfter=True
            VIn = BinSyst.GetVIn(NIJ)
            VOut = BinSyst.FreeSite.__len__() - VIn
            self.Prob = self.Prob * (InBefore * self.Pbias * VOut + (1-InBefore) * (1-self.Pbias) * VIn)
            self.Prob = self.Prob / (InAfter * self.Pbias * VOut + (1-InAfter) * (1-self.Pbias) * VIn)
        return self.Prob
    def Reverse(self):
        return self.CopySystem#System(Old_System=self.CopySystem)
    def Count(self,Success,DE=0):
        #self.DE+=abs(DE)
        if DE>0:
            self.DE+=DE
            self.DEP+=1.
        else:
            self.DEN+=1.
        if Success:
            self.Success+=1.
        else:
            self.Refuse+=1.
        if Success and DE>=0:
            self.DEPA+=1
    def MakeStat(self,time,Beta):
        Ntot=self.Success+self.Refuse
        #DEPArate=self.DEPA/Ntot
        if self.DEP!=0:
            self.DEPArate=self.DEPA/self.DEP
            self.avDE=self.DE/self.DEP
        else :
            self.avDE=0
            self.DEPArate=0
        DEPrate=self.DEP/Ntot
        DENrate=self.DEN/Ntot
        RefusalRate=self.Refuse/Ntot
        AcceptanceRate=self.Success/Ntot
        with open('Res/Sim'+str(self.SimNum)+'/Stat.out','a') as myfile:
            myfile.write(str(time)+' '+str(Beta)+' '+str(AcceptanceRate)+' '+str(RefusalRate)+' ')
            myfile.write(str(self.Nmove)+' '+str(self.radius)+'\n')
        with open('Res/Sim'+str(self.SimNum)+'/AdvanceStat.out','a') as myfile:
            myfile.write(str(time)+' '+str(Beta)+' '+str(DEPrate)+' '+str(DENrate)+' '+str(self.DEPArate))
            myfile.write(' '+str(self.avDE)+'\n')
        if AcceptanceRate > 0.6:
            self.Harder()
        elif AcceptanceRate < 0.4 :
            self.Softer()
        self.DE,self.DEN,self.DEP,self.DEPA=0,0,0,0
        self.Success=0
        self.Refuse=0
    def Harder(self):
        # we start by increasing the radius if it's not infinity
        # there are 10 steps of increasment from Np/20  to  Np/2
        # after Np/2 the radius becomes infinity
        if self.radius!=np.inf:
            if self.radius>=self.Np/2:
                self.radius=np.inf
            else :
                self.radius+=self.Np//20
        elif self.Nmove<=self.Np//10:
            #if the radius is already infinity we  multiply  the
            #number of move per step by 2
            self.Nmove+=1
    def Softer(self):
        if self.radius==self.Np//20 and self.Nmove==1:
            return
        if self.radius==np.inf and self.Nmove==1:
            self.radius=self.Np//2
        elif self.radius==np.inf :
            self.Nmove=self.Nmove//2
        elif self.radius>max(self.Np//20,3):
            self.radius-=self.Np//20
