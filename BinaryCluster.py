import numpy as np
import random as rd
import sys
import copy

class BinaryCluster:
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
                self.WindowArray[ij[0],ij[1]]=1
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
        return len([u for u in self.RealBoundarySites if u[0]>=0 and u[0]<=self.Lx and u[1]>=0 and u[1]<self.Ly])
    def ComputeBoundarySites(self):
        BoundarySet=set()
        #self.BuildOccupiedSites()
        for ij in self.OccupiedSite:
            for neigh in self.Get_Neighbors(ij,Free=True):
                BoundarySet.add(neigh)
        self.BoundarySites=np.array(list(BoundarySet),dtype=int)
        self.RealBoundarySites=copy.copy(self.BoundarySites)
        self.RealBoundarySites[:,0]+=self.Xg-self.MidX
        self.RealBoundarySites[:,1]+=self.Yg-self.MidY
        self.NBoundary=self.BoundarySites.__len__()
        # convert the array to a list of tuple
        self.BoundarySites=list(map(tuple,self.BoundarySites))
        self.RealBoundarySites = list(map(tuple,self.RealBoundarySites))
    def Get_Neighbors(self, ij,Occupied=False,Free=False):
        Res=list()
        if ij[0]+1<self.Size:
            Res.append((ij[0]+1,ij[1]))
        #elif Free:
        #    Res.append((np.infty,ij[1]))
        if ij[0]-1>=0:
            Res.append((ij[0]-1,ij[1]))
        #elif Free:
        #    Res.append((np.infty,ij[1]))
        if(ij[0]+ij[1])%2==0:
            if ij[1]+1<self.Size:
                Res.append((ij[0],ij[1]+1))
        #    elif Free:
        #        Res.append((ij[0],np.infty))
        else :
            if ij[1]-1>=0:
                Res.append((ij[0],ij[1]-1))
        #    elif Free :
        #        Res.append((ij[0],np.infty))
        if Occupied:
            for n in reversed(range(Res.__len__())):
                if self.WindowArray[Res[n]]!=1:
                    del Res[n]
        if Free:
            for n in reversed(range(Res.__len__())):
                #if all(res!=np.infty for res in Res[n]):
                if all(res>0 and res < self.Size for res in Res[n]):
                    if self.WindowArray[Res[n]]!=0:
                        del Res[n]
        Res=set(Res)
        return Res

    def CheckDiscontiguity(self,RealSpaceij=None,WindowSpaceij=None): #return true is it's discontiguous and false if it's contiguous
        if RealSpaceij:
            ij = self.RealToWindow(RealSpaceij)
        elif WindowSpaceij:
            ij = self.WindowToReal(WindowSpaceij)
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
