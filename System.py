import numpy as np
import copy
from Cluster import *
from BinaryCluster import *
from matplotlib.colors import LinearSegmentedColormap
import random as rd

cdict = {'blue':   ((0.0,  0.9,0.9),
                    (0.5,  0.4, 0.4),
                    (1.0,  0.1, 0.1)),

         'green': ((0.0,  0.5, 0.5),
                   (0.5 , 1, 1),
                   (1.0,  0.3, 0.3)),

         'alpha': ((0.0,  1, 1),
                   (0.5 , 0.8, 0.8),
                   (1.0,  1, 1)),

         'red':  ((0.0,  0.4, 0.4),
                   (0.5,  0.5, 0.5),
                   (1.0,  0.9,0.9)),
}
cm = LinearSegmentedColormap('my_colormap', cdict, 1024)
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
# This class system basically contain an array of 0 and 1. it then sort the  list  of
# 0 and 1 into a list of cluster. Each cluster is a c++ object, with its own  energy.
# the class system as few main function :
# - Make_cluster : Build a  list  of  neighbors  1,  and  create  an  object  cluster
# associated to this.
# -Make_Move : RmRandParticle + AddRandParticle + identify the affected clusters
# -RmRandParticle : Random 1->0
# -Addrandparticle : Random 0->1
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
class System:
    #There are three ways of initializing the System
    # 1- with another system -> make a Copy
    # 2- with an array of 0 and 1 -> create the cluster associated
    # 3- with nothing, we will then use the inner functions to
    # fill anything inside
    # Note :
    #-------
    # The binary system are, or a list of 0 and 1, or a list of tuple indices.
    def __init__(self,
                Lx=5,
                Ly=5,
                Eps=0.1,
                Kcoupling=1.,
                Kmain=1.,
                Kvol=1.,
                J=1.,
                Old_System=None,
                State=None):
        # Elastic constant
        self.J=J
        self.Eps=Eps
        self.Kmain=Kmain
        self.Kvol=Kvol
        self.Kcoupling=Kcoupling
        self.BinaryClusters=dict()#list() # list of object set of 0 and 1 that are neighbors
        self.ObjectClusters=dict() # list of object cluster linked witht he c++ program
        self.SiteToCluster=dict()
        self.KeyList=set(np.arange(200))
        self.OccupiedSite=set()
        self.FreeSite=set()
        # Two ways of initializing the system
        if type(State)==np.ndarray:
            self.Build_From_Array(State) # From an array
            self.Compute_Energy()
        elif Old_System!=None:
            self.Build_From_System(Old_System) # Copy a system
        # Plus the possibility to start with an empty system
        else:
            self.Lx=Lx
            self.Ly=Ly
            #State of 0 and 1
            self.State=np.array([np.zeros(self.Ly,dtype=int) for _ in range(self.Lx)])
            self.Np=0
            self.SetOccupiedAndFreeSites()
            self.Compute_Energy()
    def Build_From_System(self,Old):
        #From another System : we copy everything
        self.Lx,self.Ly=Old.Lx,Old.Ly
        self.State=copy.copy(Old.State)
        self.Kmain,self.Kvol,self.Eps,self.Kcoupling=Old.Kmain,Old.Kvol,Old.Eps,Old.Kcoupling
        self.J=Old.J
        self.Np=Old.Np
        self.ElasticEnergy=Old.ElasticEnergy
        self.SurfaceEnergy=Old.SurfaceEnergy
        self.OccupiedSite=copy.copy(Old.OccupiedSite)
        self.FreeSite=copy.copy(Old.FreeSite)
        self.SiteToCluster=copy.deepcopy(Old.SiteToCluster)
        # need to deep copy all the objects
        self.BinaryClusters=copy.deepcopy(Old.BinaryClusters)
        # need to deep copy all the object in the list
        self.ObjectClusters=dict() # create a new list
        for key,Clust in Old.ObjectClusters.items(): # take every cluster in the old system
            # Cluster has a built-in copy constructor if the old_cluster
            # argument is given
            #self.BinaryClusters[key]=copy.copy(Old.BinaryClusters[key])
            self.ObjectClusters[key] = Cluster(old_cluster=Clust)
        if self.ObjectClusters.__len__()!=Old.ObjectClusters.__len__():
                print('fail')
                self.PlotPerSite()
                Old.PlotPerSite()
    def Build_From_Array(self,State):
        # This function is an extension of the __init__ function.
        # It initialize a system on the basis of a given array of
        # 0/1.
        self.Lx=State.shape[0]
        self.Ly=State.shape[1]
        self.State=State
        self.SetOccupiedAndFreeSites()
        self.Np=self.OccupiedSite.__len__()
        self.MakeClusters()
    def PrintBinary(self):
        for j in reversed(range(self.State.shape[1])):
            for i in range(self.State.shape[0]):
                print(str(self.State[i,j])+" ",end='')
            print('\n',end='')
        print('\n',end='')
    def g_Np(self):
        return self.OccupiedSite.__len__()
    def Compute_Energy(self):
        self.ElasticEnergy=0
        self.SurfaceEnergy=0
        for Clust in self.ObjectClusters.values():
            self.ElasticEnergy+=Clust.Energy
        for Clust in self.BinaryClusters.values():
            self.SurfaceEnergy+=Clust.NBoundary*self.J
        return self.ElasticEnergy+self.SurfaceEnergy
    def __del__(self):
        for cluster in self.ObjectClusters.values():
            del(cluster)
    def SetOccupiedAndFreeSites(self):
        for i in range(self.State.shape[0]):
            for j in range(self.State.shape[1]):
                if self.State[i,j]==1 :
                    self.OccupiedSite.add((i,j))
                else :
                    self.FreeSite.add((i,j))
    def MakeClusters(self):
        # This function build the clusters as binary clusters.
        # Then from this build it create c++ object for each cluster.
        self.MakeBinaryClusters(self.OccupiedSite)
        self.MakeObjectClusters()
    def MakeObjectClusters(self):
        #Make sure we delete the object before remaking all of them
        for cluster in self.ObjectClusters.values():
            del(cluster)
        for key,BinClust in self.BinaryClusters.items():
            self.ObjectClusters[key]=Cluster(State=BinClust.WindowArray,
                                        eps=self.Eps,
                                        Kmain=self.Kmain,
                                        Kcoupling=self.Kcoupling,
                                        Kvol=self.Kvol,
                                        Xg=BinClust.Xg,
                                        Yg=BinClust.Yg)
    def MakeBinaryClusters(self,SitesNoCluster):
        # Given an array of 0/1 called self.State this function split all the
        # 1 that respect a neighboring relation (given by the function Neighbors)
        # into a list of array indices.
        Keys=set()
        while SitesNoCluster.__len__()!=0:# The process ends when every sites is in a cluster
            # Start creating a new cluster
            Cluster=set(rd.sample(SitesNoCluster,1)) # Note this will reset the clusters list
            ToIterate=Cluster # Initialize the cluster growth with the first site on which we are gonna add its neighbors
            ToAdd=set() # this will be the list of neighbors we get from iterate
            while ToIterate.__len__()!=0: # once all the new neighbors are already in the cluster, we stop
                for sites in ToIterate: # Access to all the neighbors of the sites of the cluster
                    #ToAdd.update(set([neigh for neigh in self.Get_Neighbors(sites) if neigh in self.OccupiedSite])) # Check that they are occupied
                    ToAdd.update(set([neigh for neigh in self.Get_Neighbors(sites,Occupied=True)])) # Check that they are occupied
                ToIterate=ToAdd.difference(Cluster) # Remove the one that were already in the cluster
                Cluster.update(ToAdd) # Add them in the cluster
                ToAdd=set() # reset the adding list
            Key=rd.sample(self.KeyList,1)[0]
            self.KeyList.remove(Key)
            for site in Cluster:
                self.SiteToCluster[site] = Key
            #self.BinaryClusters.append(BinaryCluster(Cluster,self.Lx,self.Ly))
            self.BinaryClusters[Key] = BinaryCluster(Cluster,self.Lx,self.Ly)
            # Remove the particles that are in the newly created cluster
            SitesNoCluster=SitesNoCluster.difference(Cluster)
            Keys.add(Key)
        return Keys
    def Neighbors(self,i1,j1,i2,j2):
        #return true if ij1 and ij2 are neighbors false otherwise
        if j1==j2: # Same line
            if i1==i2-1 or i1==i2+1 : # right or left neighbors
                return True
        elif i1==i2: # Same column
            if (i1+j1)%2==0 : # which means ij1 is down : \/
                if j1==j2-1: # ij1 is just below ij2
                    return True
            else : # which means ij1 is up : /\
                if j1==j2+1: # ij1 is just over ij2
                    return True
        return False
    def Get_Neighbors(self, ij,Occupied=False,Free=False):
        Res=list()
        if ij[0]+1<self.Lx:
            Res.append((ij[0]+1,ij[1]))
        elif Free:
            Res.append((np.infty,ij[1]))
        if ij[0]-1>=0:
            Res.append((ij[0]-1,ij[1]))
        elif Free:
            Res.append((np.infty,ij[1]))
        if(ij[0]+ij[1])%2==0:
            if ij[1]+1<self.Ly:
                Res.append((ij[0],ij[1]+1))
            elif Free:
                Res.append((ij[0],np.infty))
        else :
            if ij[1]-1>=0:
                Res.append((ij[0],ij[1]-1))
            elif Free :
                Res.append((ij[0],np.infty))
        if Occupied:
            for n in reversed(range(Res.__len__())):
                if self.State[Res[n]]!=1:
                    del Res[n]
        if Free:
            for n in reversed(range(Res.__len__())):
                if all(res!=np.infty for res in Res[n]):
                    if self.State[Res[n]]!=0:
                        del Res[n]
        Res=set(Res)
        return Res
    def AddRandParticle(self,IJ=None,Radius=np.infty):
        if IJ==None:
            IJ=(self.Lx//2,self.Ly//2)
        if Radius < min([self.Lx//2,self.Ly//2]):
            PickingSite=self.FreeSite.intersection(
                                    set((i,j) for i in range(IJ[0]-Radius,IJ[0]+Radius)
                                              for j in range(IJ[1]-Radius,IJ[1]+Radius)))
        else :
            PickingSite=self.FreeSite
        #SetToPick=set([(i,j) for i,line in enumerate(PickingSite) for j,state in enumerate(line) if state==1])
        try:
            RandomSite=rd.sample(PickingSite,1)[0]
        except ValueError:
            print("No free site available, cannot add any particle")
            return
        self.AddParticle(RandomSite)
        return RandomSite
    def AddParticle(self,IJ):
        if not isinstance(IJ,tuple):
            print('IJ isn t a tuple')
            print(IJ)
            raise ValueError
        if not IJ.__len__()==2:
            print('IJ s length isn t correct')
            print(IJ)
            raise ValueError
        self.OccupiedSite.add(IJ)
        self.FreeSite.remove(IJ)
        self.State[IJ]=1
        # Adding a particle may lead several cluster to merge.
        # Thus we delete all the concerned cluster and rebuild
        # the cluster starting from the RandomSite
        # Warning ! delete from last to first
        #-------------------------------------------------
        # Get the occupied neighbors of the added site
        Neigh=self.Get_Neighbors(IJ,Occupied=True)
        # Get the corresponding affected clusters (in a set in case there are doublet)
        Clust=self.GetAffectedCluster(Neigh)
        # Delete by reversed order.
        for AffectedCluster in reversed(sorted(Clust)):
            self.KeyList.add(AffectedCluster)
            del self.BinaryClusters[AffectedCluster]
            del self.ObjectClusters[AffectedCluster]
        # Remake the cluster that is connected to the RandomSite
        Keys = self.MakeBinaryClusters({IJ})
        # Add the last Binary cluster made (the one just before) as a c++
        # cluster object
        for key in Keys:
            #Fake loop because there can only be one Key
            self.ObjectClusters[key] = Cluster(State=self.BinaryClusters[key].WindowArray,
                                            eps=self.Eps,
                                            Kmain=self.Kmain,
                                            Kcoupling=self.Kcoupling,
                                            Kvol=self.Kvol,
                                            Xg=self.BinaryClusters[key].Xg,
                                            Yg=self.BinaryClusters[key].Yg)
    def AddParticleOutVicinity(self,NIJ):
        ClustNIJ = self.FindCluster(NIJ)
        In = True
        if self.FreeSite.__len__() <= ClustNIJ.RealBoundarySites.__len__():
            raise ValueError
        while In:
            try:
                RandomSite=rd.sample(self.FreeSite,1)[0]
            except ValueError:
                print("No free site available, cannot add any particle")
                raise
            if not self.CheckVicinity(RandomSite,NIJ):
                In = False
        self.AddParticle(RandomSite)
        return RandomSite
    def AddParticleVicinity(self,NIJ=None,Clust=None,NoFusion=False):
        if NIJ :
            ClustNIJ = self.FindCluster(NIJ)
        else :
            ClustNIJ = Clust
        OutBox=True
        if self.OccupiedSite.__len__() == self.Lx*self.Ly:
            raise ValueError
        count=0
        while OutBox:
            if ClustNIJ:
                RandomSite = rd.choice(ClustNIJ.RealBoundarySites)
                NaffectedClusterMax = 1
            else:
                RandomSite = rd.sample(self.FreeSite,1)[0]
                NaffectedClusterMax = 0
            if RandomSite[0]>=0 and RandomSite[0]<self.Lx and RandomSite[1]>=0 and RandomSite[1]<self.Ly:
                if RandomSite!=NIJ:
                    OutBox=False
            else:
                continue
            if NoFusion:
                if self.GetAffectedCluster(self.Get_Neighbors(RandomSite,Occupied=True)).__len__() >NaffectedClusterMax:
                    OutBox=True
            count+=1
            if count>=self.Lx*self.Ly*1000:
                raise ValueError
        try:
            self.AddParticle(RandomSite)
        except :
            print(RandomSite)
            print(NIJ)
            ClustNIJ.PrintBinary()
            print(ClustNIJ.RealBoundarySites)
            self.PrintBinary()
            raise
        return RandomSite
    def RemoveParticle(self,IJ):
        if not isinstance(IJ,tuple):
            print('IJ isn t a tuple')
            raise ValueError
            print(IJ)
        if not IJ.__len__()==2:
            print('IJ s length isn t correct')
            raise ValueError
            print(IJ)
        for AffectedCluster in reversed(sorted(self.GetAffectedCluster({IJ}))):
            #self.ObjectClusters[AffectedCluster].PlotPerSite()
            self.KeyList.add(AffectedCluster)
            del self.BinaryClusters[AffectedCluster]
            del self.ObjectClusters[AffectedCluster]
        #self.PlotPerSite()
        # Actualize Free/Occupied Site and State
        self.FreeSite.add(IJ)
        self.OccupiedSite.remove(IJ)
        self.State[IJ]=0
        # Remake all the binary clusters and keep track of the number of
        # Binary cluster created to know how many object cluster we need
        # to create.
        SizeBefore=self.BinaryClusters.__len__()
        Keys = self.MakeBinaryClusters(self.Get_Neighbors(IJ,Occupied=True))
        #Nclust=self.BinaryClusters.__len__()-SizeBefore
        # Make sure that the BinaryCluster[n] correspond to the ObjectClusters[n]
        #for n in range(Nclust):
            #k=Nclust-n
        for key in Keys:
            self.ObjectClusters[key] = Cluster(State=self.BinaryClusters[key].WindowArray,
                                        eps=self.Eps,
                                        Kmain=self.Kmain,
                                        Kcoupling=self.Kcoupling,
                                        Kvol=self.Kvol,
                                        Xg=self.BinaryClusters[key].Xg,
                                        Yg=self.BinaryClusters[key].Yg)
    def RemoveRandParticle(self,NIJ=False):
        # Try to remove a particle
        Same = True
        while Same:
            try :
                RandomParticle=rd.sample(self.OccupiedSite,1)[0]
            except ValueError:
                print("No particle in the system to remove")
                return
            if NIJ:
                if NIJ!=RandomParticle:
                    Same = False
            else :
                Same=False
        if NIJ:
            Vicinity = self.CheckInside(RandomParticle,NIJ)
        # Delete the only affected cluster, make a loop because GetAffectedClust
        # ers returns a set
        self.RemoveParticle(RandomParticle)
        if NIJ:
            return RandomParticle, Vicinity
        return RandomParticle
    def RmRandContiguousParticle(self,Clust=None):
        if not Clust:
            try :
                Clust = rd.choice(list(self.BinaryClusters.values()))
            except ValueError:
                print("No Cluster in the system to remove a particle from")
                return
        Destroyed=False
        if Clust.OccupiedSite.__len__() == 1:
            Destroyed=True
        ij=rd.sample(Clust.RealSpaceSites,1)[0] # remove
        while Clust.CheckDiscontiguity(RealSpaceij=ij): # check
            ij=rd.sample(Clust.RealSpaceSites,1)[0] # remove
        self.RemoveParticle(ij)
        return ij,Destroyed
    def SelectRandomNeighbor(self):
        try :
            NIJ=rd.sample(self.OccupiedSite,1)[0]
        except ValueError:
            print("No particle in the system to select")
            raise
        return NIJ
    def CheckVicinity(self,IJ,NIJ):
        # Find the cluster of NIJ and check wether IJ is in the Vicinity
        # Meaning in the cluster of NIJ
        ClustNIJ = self.FindCluster(NIJ)
        if IJ in ClustNIJ.RealBoundarySites:
            return True
        return False
    def CheckInside(self,IJ,NIJ):
        # Find the cluster of NIJ and check wether IJ is in the Vicinity
        # Meaning in the cluster of NIJ
        ClustNIJ = self.FindCluster(NIJ)
        if IJ in ClustNIJ.RealSpaceSites:
            return True
        return False
    def FindCluster(self,IJ):
        Cluster=None
        #for Clust in self.BinaryClusters:
        #    if IJ in Clust.RealSpaceSites:
        #        Cluster=Clust
        #        break
        return self.BinaryClusters[self.SiteToCluster[IJ]]
    def GetVIn(self,NIJ):
        Clust = self.FindCluster(NIJ)
        return Clust.GetVIn()
    def GetAffectedCluster(self,SiteConcerned):
        # Must return a set (to avoid doublet) of cluster indices
        AffectedCluster=set()
        for Neigh in SiteConcerned:
            if self.State[Neigh]==1:
                try:
                    AffectedCluster.add(self.SiteToCluster[Neigh])
                except KeyError:
                    print(SiteConcerned)
                    print(self.SiteToCluster)
                    print(Neigh)
                    raise
                #for n,Cluster in enumerate(self.BinaryClusters):
                #    if Neigh in Cluster.RealSpaceSites:
                #        AffectedCluster.add(n)
        return AffectedCluster
    def PlotPerSite(self,figuresize=(7,5),zoom=1):
        fig,ax=plt.subplots(figsize=figuresize)
        for objcluster in self.ObjectClusters.values():
            objcluster.PlotPerSite(show=False,zoom=zoom,ax=ax)
        ax.set_xlim([0,(self.Lx+3)/zoom])
        ax.set_ylim([0,(self.Ly+3)/zoom])
        plt.show()
    def CheckBoundaryFree(self):
        for Clust in self.BinaryClusters:
            for Boundary in Clust.RealBoundarySites:
                if Boundary[0]>=0 and Boundary[0]<self.Lx and Boundary[1]>=0 and Boundary[1]<self.Ly:
                    if self.State[Boundary]!=0:
                        print('BoundarySite aren t free')
                        raise ValueError
    def CheckClusterToSite(self):
        for Clust in self.BinaryClusters.values():
            for Site in Clust.RealSpaceSites:
                if not Site in list(self.SiteToCluster.keys()):
                    print('a site in cluster isn t in sitetocluster')
                    raise KeyError        
        for Site,Clust in self.SiteToCluster.items():
            if self.State[Site]==1:
                if not Site in self.BinaryClusters[Clust].RealSpaceSites:
                    print('a site isn t in the cluster')
                    print(self.SiteToCluster)
                    print(self.BinaryClusters)
                    self.PlotPerSite()
                    raise KeyError

        for key in self.BinaryClusters.keys():
            for i in range(self.BinaryClusters[key].WindowArray.shape[0]):
                for j in range(self.BinaryClusters[key].WindowArray.shape[1]):
                    if self.BinaryClusters[key].WindowArray[i,j] != self.ObjectClusters[key].state[i,j]:
                        print('the two clusters does not correspong')
                        raise KeyError
    def PrintPerSite(self,FileName='Noname.txt',Path=''):
        XY=[]
        for objcluster in self.ObjectClusters.values():
            XY.extend(objcluster.PlotPerSite(show=False,ToPrint=True,Path=Path))
        np.savetxt(FileName,XY)
