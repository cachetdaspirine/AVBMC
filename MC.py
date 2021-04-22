import numpy as np
import pathlib

CouplingRelation=np.load(str(pathlib.Path(__file__).parent.absolute())+'/CouplingRelation.npy',allow_pickle=True)
MainRelation=np.load(str(pathlib.Path(__file__).parent.absolute())+'/MainRelation.npy',allow_pickle=True)
VolumiqueRelationP=np.load(str(pathlib.Path(__file__).parent.absolute())+'/VolumiqueRelationP.npy',allow_pickle=True)
VolumiqueRelationM=np.load(str(pathlib.Path(__file__).parent.absolute())+'/VolumiqueRelationM.npy',allow_pickle=True)
#print(VolumiqueRelationP)
#print(VolumiqueRelationM)

def get_Mc(k=0,kc=0,eps=0,kA=0,Parameter = False):
    if Parameter:
        k,kc,eps,kA = Parameter.k, Parameter.kc,Parameter.epsilon,Parameter.kA
    Mc = np.array([np.zeros(12,dtype=float) for _ in range(12)])
    Ap = (1+eps)**2*3**0.5/4.
    Am = (1-eps)**2*3**0.5/4.
    for R in CouplingRelation:
        Mc[R[0]]+=R[1]*kc/2.
    for R in MainRelation:
        Mc[R[0]]+=R[1]*k/2.
    for R in VolumiqueRelationP:
        Mc[R[0]]+=R[1]*kA/(2*Ap)
    for R in VolumiqueRelationM:
        Mc[R[0]]+=R[1]*kA/(2*Am)
    # Mc[0,0]+=3./4.
    # Mc[0,1]+=3**0.5/2.
    # Mc[1,1]+=1./4.
    # Mc[0,4]+=-3./2.
    # Mc[1,4]+=-3**0.5/2.
    # Mc[4,4]+=3./4.
    # Mc[0,5]+=-3**0.5/2.
    # Mc[1,5]+=-1./2.
    # Mc[4,5]+=3**0.5/2.
    # Mc[5,5]+=1./4.
#
    # Mc[1,1]+=1
    # Mc[1,9]+=-2
    # Mc[9,9]+=1
#
    # Mc[4,4]+=3./4.
    # Mc[4,5]+=-3**0.5/2.
    # Mc[5,5]+=1./4.
    # Mc[4,8]+=-3./2.
    # Mc[5,8]+=3**0.5/2.
    # Mc[8,8]+=3./4.
    # Mc[4,9]+=3**0.5/2.
    # Mc[5,9]+=-1./2.
    # Mc[8,9]+=-3**0.5/2.
    # Mc[9,9]+=1./4.
#
    # Mc[3,3]+=1
    # Mc[3,7]+=-2
    # Mc[7,7]+=1
#
    # Mc[10,10]+=3./4.
    # Mc[10,11]+=-3**0.5/2.
    # Mc[11,11]+=1./4.
    # Mc[10,2]+=-3./2.
    # Mc[11,2]+=3**0.5/2.
    # Mc[2,2]+=3./4.
    # Mc[10,3]+=3**0.5/2.
    # Mc[11,3]+=-1./2.
    # Mc[2,3]+=-3**0.5/2.
    # Mc[3,3]+=1./4.
#
    # Mc[10,10]+=3./4.
    # Mc[10,11]+=3**0.5/2.
    # Mc[11,11]+=1./4.
    # Mc[10,6]+=-3./2.
    # Mc[11,6]+=-3**0.5/2.
    # Mc[6,6]+=3./4.
    # Mc[10,7]+=-3**0.5/2.
    # Mc[11,7]+=-1./2.
    # Mc[6,7]+=3**0.5/2.
    # Mc[7,7]+=1./4.
#
#
    # Mc=k/2.*Mc
#
    # Mc[0,0]+=5./4.*kappa/2.
    # Mc[2,2]+=5./4.*kappa/2.
    # Mc[6,6]+=5./4.*kappa/2.
    # Mc[8,8]+=5./4.*kappa/2.
#
    # Mc[1,1]+=3./4.*kappa/2.
    # Mc[3,4]+=3./4.*kappa/2.
    # Mc[7,7]+=3./4.*kappa/2.
    # Mc[9,9]+=3./4.*kappa/2.
#
    # Mc[4,4]+=0.5*kappa/2.
    # Mc[10,10]+=0.5*kappa/2.
#
    # Mc[5,5]+=3./2.*kappa/2.
    # Mc[1,1]+=3./2.*kappa/2.
#
    # Mc[0,2]+=-2.*kappa/2.
    # Mc[6,8]+=-2*kappa/2.
    # Mc[2,3]+=3**0.5/2.*kappa/2.
    # Mc[2,4]+=-0.5*kappa/2.
    # Mc[3,4]+=-3**0.5/2.*kappa/2.
    # Mc[2,5]+=-3**0.5/2.*kappa/2.
    # Mc[3,5]+=-3./2.*kappa/2.
#
    # Mc[4,6]+=-0.5*kappa/2.
    # Mc[5,6]+=3**0.5/2.*kappa/2.
    # Mc[4,7]+=3**0.5/2.*kappa/2.
    # Mc[5,7]+=-3**0.5/2.*kappa/2.
    # Mc[6,7]+=-3**0.5/2.*kappa/2.
#
    # Mc[10,8]+=0.5*kappa/2.
    # Mc[11,8]+=3**0.5/2.*kappa/2.
    # Mc[10,9]+=-3**0.5/2.*kappa/2.
    # Mc[11,9]+=-3./2.*kappa/2.
    # Mc[8,9]+=3**0.5/2.*kappa/2.
#
    # Mc[0,1]+=-3**0.5/2.*kappa/2.
    # Mc[0,10]+=-0.5*kappa/2.
    # Mc[1,10]+=3**0.5/2.*kappa/2.
    # Mc[0,11]+=3**0.5/2.*kappa/2.
    # Mc[1,11]+=-3/2.*kappa/2.
#
    # Ap = (1+eps)**2*3**0.5/4.
    # Am = (1-eps)**2*3**0.5/4.
    #Volumique energy :
    #A+
    # Mc[0,0]+=1./16. * kA/2./Ap
    # Mc[1,1]+=3./16. * kA/2./Ap
    # Mc[4,4]+=1./4. * kA/2./Ap
    # Mc[8,8]+=1./16. * kA/2./Ap
    # Mc[9,9]+=3./16. * kA/2./Ap
    #A-
    # Mc[10,10]+=1./4. * kA/2./Am
    # Mc[3,3]+=3./16. * kA/2./Am
    # Mc[2,2]+=1./16. * kA/2./Am
    # Mc[6,6]+=1./16. * kA/2./Am
    # Mc[7,7]+=3/16. * kA/2./Am
#
    #A+
    # Mc[0,1]+=3**0.5/8. * kA/2./Ap
    # Mc[0,4]+=-1./4. * kA/2./Ap
    # Mc[1,4]+=-3**0.5/4 * kA/2./Ap
    # Mc[8,9]+= -3**0.5/8. * kA/2./Ap
    # Mc[1,8]+=3**0.5/8. * kA/2./Ap
    # Mc[1,9]+=-3./8. * kA/2./Ap
    # Mc[0,8]+=1./8. * kA/2./Ap
    # Mc[4,8]+=-1./4. * kA/2./Ap
    # Mc[0,9] +=-3**0.5/8. * kA/2./Ap
    # Mc[4,9]+=3**0.5/4. * kA/2./Ap
#
    #A-
    # Mc[10,2]+=-1./4. * kA/2./Am
    # Mc[10,3]+=3**0.5/4. * kA/2./Am
    # Mc[2,3]+=-3**0.5/8. * kA/2./Am
    # Mc[10,6]+=-1./4. * kA/2./Am
    # Mc[2,6]+=1./8. * kA/2./Am
    # Mc[3,6]+=-3**0.5/8. * kA/2./Am
    # Mc[10,7]+=-3**0.5/4.* kA/2./Am




    Mc = (Mc+np.transpose(Mc))
    q0 = np.array([(1+eps)/(2*3**0.5),  #0
                    (1+eps)/2.,          #1
                    -(1-eps)/(2*3**0.5), #2
                    (1-eps)/2.,          #3
                    -(1+eps)/3**0.5,     #4
                    0.,                  #5
                    -(1-eps)/(2*3**0.5), #6
                    -(1-eps)/2.,         #7
                    (1+eps)/(2*3**0.5),  #8
                    -(1+eps)/2,          #9
                    (1-eps)/(3**0.5),    #10
                    0.])                 #11
    return Mc,q0




qregular = np.array([(1)/(2*3**0.5),
               (1)/2.,
               -(1)/(2*3**0.5),
               (1)/2.,
               -(1)/3**0.5,
               0.,
               -(1)/(2*3**0.5),
               -(1)/2.,
               (1)/(2*3**0.5),
               -(1)/2,
               (1)/(3**0.5),
               0.])
