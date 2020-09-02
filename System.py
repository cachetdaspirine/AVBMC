import numpy as np
from Cluster import *
from BinaryCluster import *
from copy import deepcopy
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
        self.BinaryClusters=list() # list of object set of 0 and 1 that are neighbors
        self.ObjectClusters=list() # list of object cluster linked witht he c++ program
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
        # need to deep copy all the objects
        self.BinaryClusters=deepcopy(Old.BinaryClusters)
        # need to deep copy all the object in the list
        self.ObjectClusters=list() # create a new list
        for Clust in Old.ObjectClusters: # take every cluster in the old system
            # Cluster has a built-in copy constructor if the old_cluster
            # argument is given
            self.ObjectClusters.append(Cluster(old_cluster=Clust))
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
        for Clust in self.ObjectClusters:
            self.ElasticEnergy+=Clust.Energy
        for Clust in self.BinaryClusters:
            self.SurfaceEnergy+=Clust.NBoundary*self.J
        return self.ElasticEnergy+self.SurfaceEnergy
    def __del__(self):
        for cluster in self.ObjectClusters:
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
        for cluster in self.ObjectClusters:
            del(cluster)
        for BinClust in self.BinaryClusters:
            self.ObjectClusters.append(Cluster(State=BinClust.WindowArray,
                                        eps=self.Eps,
                                        Kmain=self.Kmain,
                                        Kcoupling=self.Kcoupling,
                                        Kvol=self.Kvol,
                                        Xg=BinClust.Xg,
                                        Yg=BinClust.Yg))
    def MakeBinaryClusters(self,SitesNoCluster):
        # Given an array of 0/1 called self.State this function split all the
        # 1 that respect a neighboring relation (given by the function Neighbors)
        # into a list of array indices.
        while SitesNoCluster.__len__()!=0:# The process ends when every sites is in a cluster
            # Start creating a new cluster
            Cluster=set(rd.sample(SitesNoCluster,1)) # Note this will reset the clusters list
            ToIterate=Cluster # Initialize the cluster growth with the first site on which we are gonna add its neighbors
            ToAdd=set() # this will be the list of neighbors we get from iterate
            while ToIterate.__len__()!=0: # once all the new neighbors are already in the cluster, we stop
                for sites in ToIterate: # Access to all the neighbors of the sites of the cluster
                    ToAdd.update(set([neigh for neigh in self.Get_Neighbors(sites) if neigh in self.OccupiedSite])) # Check that they are occupied
                ToIterate=ToAdd.difference(Cluster) # Remove the one that were already in the cluster
                Cluster.update(ToAdd) # Add them in the cluster
                ToAdd=set() # reset the adding list
            self.BinaryClusters.append(BinaryCluster(Cluster,self.Lx,self.Ly))
            # Remove the particles that are in the newly created cluster
            SitesNoCluster=SitesNoCluster.difference(Cluster)
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
        self.OccupiedSite.add(RandomSite)
        self.FreeSite.remove(RandomSite)
        self.State[RandomSite]=1
        # Adding a particle may lead several cluster to merge.
        # Thus we delete all the concerned cluster and rebuild
        # the cluster starting from the RandomSite
        # Warning ! delete from last to first
        #-------------------------------------------------
        # Get the occupied neighbors of the added site
        Neigh=self.Get_Neighbors(RandomSite,Occupied=True)
        # Get the corresponding affected clusters (in a set in case there are doublet)
        Clust=self.GetAffectedCluster(Neigh)
        # Delete by reversed order.
        for AffectedCluster in reversed(sorted(Clust)):
            del self.BinaryClusters[AffectedCluster]
            del self.ObjectClusters[AffectedCluster]
        # Remake the cluster that is connected to the RandomSite
        self.MakeBinaryClusters({RandomSite})
        # Add the last Binary cluster made (the one just before) as a c++
        # cluster object
        self.ObjectClusters.append(Cluster(State=self.BinaryClusters[-1].WindowArray,
                                    eps=self.Eps,
                                    Kmain=self.Kmain,
                                    Kcoupling=self.Kcoupling,
                                    Kvol=self.Kvol,
                                    Xg=self.BinaryClusters[-1].Xg,
                                    Yg=self.BinaryClusters[-1].Yg))
        return RandomSite
    def RemoveRandParticle(self):
        # Try to remove a particle
        try :
            RandomParticle=rd.sample(self.OccupiedSite,1)[0]
        except ValueError:
            print("No particle in the system to remove")
            return
        # Delete the only affected cluster, make a loop because GetAffectedClust
        # ers returns a set
        for AffectedCluster in reversed(sorted(self.GetAffectedCluster({RandomParticle}))):
            #self.ObjectClusters[AffectedCluster].PlotPerSite()
            del self.BinaryClusters[AffectedCluster]
            del self.ObjectClusters[AffectedCluster]
        #self.PlotPerSite()
        # Actualize Free/Occupied Site and State
        self.FreeSite.add(RandomParticle)
        self.OccupiedSite.remove(RandomParticle)
        self.State[RandomParticle]=0
        # Remake all the binary clusters and keep track of the number of
        # Binary cluster created to know how many object cluster we need
        # to create.
        SizeBefore=self.BinaryClusters.__len__()
        self.MakeBinaryClusters(self.Get_Neighbors(RandomParticle,Occupied=True))
        Nclust=self.BinaryClusters.__len__()-SizeBefore
        # Make sure that the BinaryCluster[n] correspond to the ObjectClusters[n]
        for n in range(Nclust):
            k=Nclust-n
            self.ObjectClusters.append(Cluster(State=self.BinaryClusters[-k].WindowArray,
                                        eps=self.Eps,
                                        Kmain=self.Kmain,
                                        Kcoupling=self.Kcoupling,
                                        Kvol=self.Kvol,
                                        Xg=self.BinaryClusters[-k].Xg,
                                        Yg=self.BinaryClusters[-k].Yg))
        return RandomParticle
    def GetAffectedCluster(self,SiteConcerned):
        # Must return a set (to avoid doublet) of cluster indices
        AffectedCluster=set()
        for Neigh in SiteConcerned:
            if self.State[Neigh]==1:
                for n,Cluster in enumerate(self.BinaryClusters):
                    if Neigh in Cluster.RealSpaceSites:
                        AffectedCluster.add(n)
        return AffectedCluster
    def PlotPerSite(self,figuresize=(7,5),zoom=1):
        fig,ax=plt.subplots(figsize=figuresize)
        for objcluster in self.ObjectClusters:
            objcluster.PlotPerSite(show=False,zoom=zoom,ax=ax)
        ax.set_xlim([0,self.Lx+3])
        ax.set_ylim([0,self.Ly+3])
        plt.show()
    def PrintPerSite(self,FileName='Noname.txt',Path=''):
        XY=[]
        for objcluster in self.ObjectClusters:
            XY.extend(objcluster.PlotPerSite(show=False,ToPrint=True,Path=Path))
        np.savetxt(FileName,XY)
