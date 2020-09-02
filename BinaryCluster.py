import numpy as np
import sys
import copy

class BinaryCluster:
    def __init__(self,Sites,Lx,Ly):
        # Keep track of where the sites are located in the real system
        # RealSpaceSites is a list of tuple (i,j) which represent the
        # location of each particle
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
    def BuildArray(self):
        self.ComputeCenter()
        #Build a square window of size NP*NP to be sure that the aggregate
        # fit in it. It is full of 0 for now
        self.WindowArray=np.array([np.zeros(self.Size,dtype=int)
                            for _ in range(self.Size)])
        MidX,MidY=self.Size//2, self.Size//2#middle of my aggregate
        # make sure that the aggregate is in the central sites
        # has the same orientation in the real/window space
        if(self.Xg+self.Yg)%2==1:
            MidY+=1
        self.WindowSpaceSites=set()
        for ij in self.RealSpaceSites:
            self.WindowSpaceSites.add((ij[0]-self.Xg+MidX,ij[1]-self.Yg+MidY))
        for ij in self.WindowSpaceSites:
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
    def ComputeBoundarySites(self):
        BoundarySites=set()
        self.BuildOccupiedSites()
        for ij in self.OccupiedSite:
            for neigh in self.Get_Neighbors(ij,Free=True):
                BoundarySites.add(neigh)
        self.NBoundary=BoundarySites.__len__()
    def Get_Neighbors(self, ij,Occupied=False,Free=False):
        Res=list()
        if ij[0]+1<self.Size:
            Res.append((ij[0]+1,ij[1]))
        elif Free:
            Res.append((np.infty,ij[1]))
        if ij[0]-1>=0:
            Res.append((ij[0]-1,ij[1]))
        elif Free:
            Res.append((np.infty,ij[1]))
        if(ij[0]+ij[1])%2==0:
            if ij[1]+1<self.Size:
                Res.append((ij[0],ij[1]+1))
            elif Free:
                Res.append((ij[0],np.infty))
        else :
            if ij[1]>=0:
                Res.append((ij[0],ij[1]-1))
            elif Free :
                Res.append((ij[0],np.infty))
        if Occupied:
            for n in reversed(range(Res.__len__())):
                if self.WindowArray[Res[n]]!=1:
                    del Res[n]
        if Free:
            for n in reversed(range(Res.__len__())):
                if all(res!=np.infty for res in Res[n]):
                    if self.WindowArray[Res[n]]!=0:
                        del Res[n]
        Res=set(Res)
        return Res
    def BuildOccupiedSites(self):
        self.OccupiedSite=set()
        for i,line in enumerate(self.WindowArray):
            for j,site in enumerate(line):
                if site==1:
                    self.OccupiedSite.add((i,j))
