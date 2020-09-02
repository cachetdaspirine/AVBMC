import numpy as np
import random as rd
from System import *
import os

class MonteCarlo:
    def __init__(self,Np=1,SimNum=0):
        self.Success=0
        self.Refuse=0
        self.DEP,self.DEN=0,0
        self.DEPA=0 #number of step with DE positiv and accepted
        self.DE=0
        self.radius=np.inf
        self.Nmove=10
        self.Moved=list()
        self.Np=Np
        self.SimNum=SimNum
        self.CopySystem=System()
        with open('Res/Sim'+str(self.SimNum)+'/Stat.out','w') as myfile:
            myfile.write('time Beta AcceptanceRate RefusalRate Nmove Radius\n')
        with open('Res/Sim'+str(self.SimNum)+'/AdvanceStat.out','w') as myfile:
            myfile.write('time Beta PositiveDERate NegativeDERate AcceptedPositiveDERate averageDE\n')
    def McMove(self,BinSyst):
        self.Moved.clear()
        self.CopySystem=System(Old_System=BinSyst)
        for _ in range(self.Nmove):
            IJ0=list(BinSyst.RemoveRandParticle())
            IJ1=list(BinSyst.AddRandParticle(IJ0,self.radius))
            self.Moved.append(IJ0+IJ1)
    def Reverse(self):
        return System(Old_System=self.CopySystem)
    def Count(self,Success,DE=0):
        #self.DE+=abs(DE)
        if DE>0:
            self.DE+=DE
            self.DEP+=1.
        else:
            self.DEN+=1.
        if Success:
            if self.Moved[0][0]!=self.Moved[0][2] or self.Moved[0][1]!=self.Moved[0][3]:
                self.Success+=1.
            else :
                self.Refuse+=1.
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
