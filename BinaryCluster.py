import numpy as np
import random as rd
import sys
import copy

class BinaryCluster:
    TopologieUp=list()
    TopologieDown=list()
    def __init__(self,Sites,Lx,Ly):
        # Keep track of where the sites are located in the real system
        # RealSpaceSites is a list of tuple (i,j) which represent the
        # location of each particle
        self.Lx=int(Lx)
        self.Ly=int(Ly)
        self.RealSpaceSites=Sites
        #self.ShiftSites(Lx,Ly)
        # this define the size of the box in which we are inserting this cluster
        #self.Size=max([self.RealSpaceSites.__len__()+2,5])
        self.Size=self.GetMaximumExtension()+2
        # Build an array of 0/1 in a box where there are only the one we gave as Sites
        Building=False
        while not Building:
            try :
                self.BuildArray()
            except IndexError :
                self.Size+=1
            else:
                Building=True
        self.ComputeBoundarySites()
    #------------------------------------------------------------------
    # The following function would eventually be usefull  for periodic
    # Boundary conditions
    #------------------------------------------------------------------
    # This function shift the sites in order to avoir them to cross
    # The boundary of the system
    #def ShiftSites(self,Lx,Ly):
    #    XShifted=[sites[0] for sites in self.RealSpaceSites]
    #    YShifted=[sites[1] for sites in self.RealSpaceSites]
    #    while Lx
    #Build An array of 0/1 in a smaller window
    #------------------------------------------------------------------
    def GetMaximumExtension(self):
        Xs=np.transpose(list(self.RealSpaceSites))[0]
        Ys=np.transpose(list(self.RealSpaceSites))[1]
        Xextension=max(Xs)-min(Xs)+2
        Yextension=max(Ys)-min(Ys)+2
        return max([Xextension,Yextension])
    def PrintBinary(self):
        for j in reversed(range(self.WindowArray.shape[1])):
            for i in range(self.WindowArray.shape[0]):
                print(str(self.WindowArray[i,j])+" ",end='')
            print('\n',end='')
    def PrintBoundary(self):
        for j in reversed(range(self.WindowArray.shape[1])):
            for i in range(self.WindowArray.shape[0]):
                #print(str(self.WindowArray[i,j])+" ",end='')
                if (i,j) in self.BoundarySites:
                    print(str(1)+" ",end='')
                else:
                    print(str(0)+" ",end='')
            print('\n',end='')
    def WindowToReal(self,ij):
        return (ij[0]+self.Xg-self.MidX,ij[1]+self.Yg-self.MidY)
    def RealToWindow(self,ij):
        return (ij[0]-self.Xg+self.MidX,ij[1]-self.Yg+self.MidY)
    def BuildArray(self):
        self.ComputeCenter()
        #Build a square window of size NP*NP to be sure that the aggregate
        # fit in it. It is full of 0 for now
        self.WindowArray=np.array([np.zeros(self.Size,dtype=int)
                            for _ in range(self.Size)])
        self.MidX,self.MidY=self.Size//2, self.Size//2#middle of my aggregate
        # make sure that the aggregate is in the central sites
        # has the same orientation in the real/window space
        if(self.Xg+self.Yg)%2==1:
            self.MidY+=1
        OccupiedList=list()
        for ij in self.RealSpaceSites:
            #self.OccupiedSite.add((ij[0]-self.Xg+self.MidX,ij[1]-self.Yg+self.MidY))
            OccupiedList.append(self.RealToWindow(ij))
        self.OccupiedSite=list(map(tuple,OccupiedList))
        for ij in self.OccupiedSite:
            try:
                if ij[0]>=0 and ij[1]>=0:
                    self.WindowArray[ij[0],ij[1]]=1
                else :
                    raise IndexError
            except IndexError:
                raise
    # Given a list of indices in the real space domaine of occupied SitesIndices
    # this function compute the center of mass of the object in order to rebuild
    # the aggregate in a smaller window
    def ComputeCenter(self):
        self.Xg,self.Yg=0,0
        for ij in self.RealSpaceSites:
            self.Xg+=float(ij[0])
            self.Yg+=float(ij[1])
        self.Xg=int(self.Xg/len(self.RealSpaceSites)+1)
        self.Yg=int(self.Yg/len(self.RealSpaceSites)+1)
    def GetVIn(self):
        return len(self.RealBoundarySites)#len([u for u in self.RealBoundarySites])
    def ComputeBoundarySites(self):
        BoundarySet=set()
        RealBoundarySet=set()
        self.NBoundary=0
        for ij in self.OccupiedSite:
            print(self.Get_Neighbors(ij,Free=True,Border=True))
            self.NBoundary+=self.Get_Neighbors(ij,Free=True,Border=True).__len__()
            for neigh in self.Get_Neighbors(ij,Free=True):
                BoundarySet.add(neigh)
        for ij in self.OccupiedSite:
            for neigh in self.Get_Neighbors(ij,Free=True,Border=True):
                Rneigh = self.WindowToReal(neigh)
                if Rneigh[0] <self.Lx and Rneigh[0]>=0 and Rneigh[1] < self.Ly and Rneigh[1] >=0:
                    RealBoundarySet.add(Rneigh)
        self.RealBoundarySites = list(RealBoundarySet)
        self.BoundarySites = list(BoundarySet)
        #self.NBoundary = self.Get_Neighbors(ij,Border=True).__len__()
    def Get_Neighbors(self, ij,Occupied=False,Free=False,Border=False):
        # get the real or window index
        # Choose the topologie to use depending on the up/down
        if (ij[0]+ij[1])%2==0:
            Res = np.array(self.TopologieDown)+np.array(ij)
        else :
            Res = np.array(self.TopologieUp)+np.array(ij)
        # regularize the result array with only the value that can be inside the state
        if not Border:
            Resreg=np.delete(Res,np.argwhere((Res[:,0]>=self.Size) | (Res[:,0]<0) | (Res[:,1]>=self.Size) | (Res[:,1]<0)),0)
        else:
            Resreg=Res
        #Build a numpy array of tuple
        Resbis=np.empty(Resreg.__len__(),dtype=object)
        Resbis[:] = list(zip(Resreg[:,0],Resreg[:,1]))
        if Border:
            Resbis = list(Resbis)
        #check the occupancie or not
        if Border :
            for n in reversed(range(Resbis.__len__())):
                try:
                    if self.WindowArray[Resbis[n]]!=0:
                        del Resbis[n]
                except IndexError:
                    continue
        else :
            if Occupied:
                Resbis=Resbis[np.array([self.WindowArray[r]==1 for r in Resbis ])]
            elif Free:
                Resbis = Resbis[np.array([self.WindowArray[r]==0 for r in Resbis])]

        return set(Resbis)
    def CheckDiscontiguity(self,RealSpaceij=None): #return true is it's discontiguous and false if it's contiguous
        ij = self.RealToWindow(RealSpaceij)
        if len(self.Get_Neighbors(ij,Occupied=True))<=1 :
            return False
        Neigh1 = rd.sample(self.Get_Neighbors(ij,Occupied=True),1)[0]
        for Neigh2 in self.Get_Neighbors(ij,Occupied=True):
            if Neigh1!=Neigh2:
                if not self.Linked(ij,Neigh1,Neigh2):
                    return True
        return False
    def Linked(self,Changed,ij1,ij2):
        Clust={ij1}
        VectClust=[ij1]
        k=0
        while k!= len(Clust):
            for Neigh in self.Get_Neighbors(VectClust[k],Occupied=True):
                if Neigh==ij2:
                    return True
                elif Neigh!=Changed and Neigh not in Clust:
                    Clust.add(Neigh)
                    VectClust.append(Neigh)
            k+=1
        return False
    def BuildOccupiedSites(self):
        OccupiedList=list()
        for i,line in enumerate(self.WindowArray):
            for j,site in enumerate(line):
                if site==1:
                    OccupiedList.append((i,j))
        self.OccupiedSite=np.array(OccupiedList,dtype=int)
