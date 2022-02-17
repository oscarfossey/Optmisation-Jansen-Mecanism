# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:13:58 2019

@author: 2018-0560
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import openturns as ot
# from numba import jit, cuda

##############################################################################################################################################################    
#FONCTIONS INTERMEDIAIRES
##############################################################################################################################################################    

# @jit(target ="cuda")
def Mecanisme_jambe (nominal, ecartype, nmc, rotationIncrements):
    """
    Simulation du mécanisme de Jansen
    
    Parameters
    ----------
    nominal : Liste de réels
        Longueurs nominales théoriques des barres qui composent la jambe 
    ecartype : Liste de réels
        Ecart Type pour la variation de longueur pour chaque barre
    nmc : Entier
        Nombre de système simulés
    rotationIncrements : Entier
        Nombre d'incrémentation pour la simulation
        Plus ce nombre est grand plus la simulation sera précise mais le temps de calcul sera augmenté

    Returns
    -------
    SolutionX : 
    Liste de réels
        Coordonnée X du pied du mécanisme 
        
    SolutionY : 
    Liste de réels
        Coordonnée Y du pied du mécanisme 

    """
    #2 dictionaries to store the joints' X and Y coordinates
    xdict={}
    ydict={}
    bar={} # Bar lengths
    xcenter={} # center of rotation of each mech
    ycenter={}
    frameX={} # connections' distances from the center of rotation
    frameY={}
    
    
    
    
    scale_jansen = 1.5
    high = 0
    low = 1
    left = 2
    right = 3
    
    loop_count = rotationIncrements*8
    Jansen_Shift_dn = -1.05
    
    #nominal = np.array([1.5,3.8,4.15,3.93,4.01,5.58,3.94,3.67,6.57,4.9,5,6.19,0.78])
    
    #ecartype = np.array([0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03])
    #ecartype = np.array([0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002])
    sample = ot.Normal(nominal, ecartype, ot.CorrelationMatrix(13)).getSample(nmc)
    
    for mech in range(nmc):
        xdict[mech]={}
        ydict[mech]={}
        #bar[mech]={}
        for i in range(13):
            bar[mech,i+1]=sample[mech,i]*scale_jansen
        xcenter[mech]=10
        ycenter[mech]=10   
        frameX[mech]= bar[mech,2]
        frameY[mech]= bar[mech,13]
    
        
    
    #Circle algo calcs for X and Y coordinates     
    def jnt_x(ax, ay, bx, by, al, bl):
        dist = ((ax-bx)**2 + (ay-by)**2)**.5
        sidea = (al**2 - bl**2 + dist**2)/2/dist
        if al - sidea > 0:
            height = (al**2 - sidea**2)**.5
        else:
            height = 0
        Dpointx = (ax+sidea*(bx-ax)/dist)
        x1 = Dpointx + height*(ay-by)/dist
        x2 = Dpointx-height*(ay-by)/dist
        return x1,x2         
    
    def jnt_y(ax, ay, bx, by, al, bl):
        dist = ((ax-bx)**2 + (ay-by)**2)**.5
        sidea = (al**2 - bl**2 + dist**2)/2/dist
        if al - sidea > 0:
            height = (al**2 - sidea**2)**.5
        else:
            height = 0
        Dpointy = (ay+sidea*(by-ay)/dist)
        y1 = Dpointy - height*(ax-bx)/dist
        y2 = Dpointy+height*(ax-bx)/dist
        return y1,y2    
    
    def circle_algo(mech,j1,j2,b1,b2,i,solution):    
        x1,x2=jnt_x(xdict[mech,j1,i], ydict[mech,j1,i], xdict[mech,j2,i], ydict[mech,j2,i], bar[mech,b1], bar[mech,b2])
        y1,y2=jnt_y(xdict[mech,j1,i], ydict[mech,j1,i], xdict[mech,j2,i], ydict[mech,j2,i], bar[mech,b1], bar[mech,b2])
      #  print x1,y1
      #  print x2,y2
        if solution==high:
            if y1>y2:
                return x1,y1
            else:
                return x2,y2
        elif solution==low:
            if y1<y2:
                return x1,y1
            else:
                return x2,y2
        elif solution==right:
            if x1>x2:
                return x1,y1
            else:
                return x2,y2
        elif solution==left:
            if x1<x2:
                return x1,y1
            else:
                return x2,y2
    
    
    def calc_joints(): 
    
        for mech in range(nmc):
            for i in range(rotationIncrements):
                theta = ((i+30) / (rotationIncrements - 0.0)) * 2 * math.pi #creates rotationIncrements angles from 0 - 2pi
    
                joint=1
                xdict[mech,joint,i]=xcenter[mech]*scale_jansen
                ydict[mech,joint,i]=ycenter[mech]*scale_jansen+Jansen_Shift_dn
    
                joint=2
                xdict[mech,joint,i]=xcenter[mech]*scale_jansen- frameX[mech]
                ydict[mech,joint,i]=ycenter[mech]*scale_jansen+Jansen_Shift_dn- frameY[mech]
    
                joint=3
                xdict[mech,joint,i]=xcenter[mech]*scale_jansen+bar[mech,1]*np.cos(theta)
                ydict[mech,joint,i]=ycenter[mech]*scale_jansen+bar[mech,1]*np.sin(theta)+Jansen_Shift_dn
        
                joint=4
                j1=2
                j2=3
                b1=3
                b2=11        
                xdict[mech,joint,i], ydict[mech,joint,i]=circle_algo(mech,j1,j2,b1,b2,i,high)
    
                joint=5
                j1=4
                j2=2
                b1=6
                b2=5
                xdict[mech,joint,i], ydict[mech,joint,i]=circle_algo(mech,j1,j2,b1,b2,i,left)
            
                joint=6
                j1=3
                j2=2
                b1=12
                b2=4
                xdict[mech,joint,i], ydict[mech,joint,i]=circle_algo(mech,j1,j2,b1,b2,i,low)
    
                joint=7
                j1=5
                j2=6
                b1=7
                b2=8
                xdict[mech,joint,i], ydict[mech,joint,i]=circle_algo(mech,j1,j2,b1,b2,i,low)        
    
                joint=8
                j1=6
                j2=7
                b1=10
                b2=9
                xdict[mech,joint,i], ydict[mech,joint,i]=circle_algo(mech,j1,j2,b1,b2,i,low)
            
                
                joint=102
                xdict[mech,joint,i]=(xcenter[mech] + frameX[mech])*scale_jansen
                ydict[mech,joint,i]=(ycenter[mech] - frameY[mech])*scale_jansen+Jansen_Shift_dn
                   
                joint=104
                j1=102
                j2=3
                b1=3
                b2=11   
                xdict[mech,joint,i], ydict[mech,joint,i]=circle_algo(mech,j1,j2,b1,b2,i,high)
    
                joint=105
                j1=104
                j2=102
                b1=6
                b2=5
                xdict[mech,joint,i], ydict[mech,joint,i]=circle_algo(mech,j1,j2,b1,b2,i,right)
    
                joint=106
                j1=3
                j2=102
                b1=12
                b2=4
                xdict[mech,joint,i], ydict[mech,joint,i]=circle_algo(mech,j1,j2,b1,b2,i,low)
    
                joint=107
                j1=105
                j2=106
                b1=7
                b2=8
                xdict[mech,joint,i], ydict[mech,joint,i]=circle_algo(mech,j1,j2,b1,b2,i,low)
    
                joint=108
                j1=106
                j2=107
                b1=10
                b2=9
                xdict[mech,joint,i], ydict[mech,joint,i]=circle_algo(mech,j1,j2,b1,b2,i,low)
                
    
    
    calc_joints()
    
    
    
    """
    couleur=['w','b', 'g', 'r', 'b', 'r', 'y', 'k']
    
    fig = plt.figure()  
    
    for mech in range(nmc):
        for i in range(8):
            for k in range(rotationIncrements-1):
                plt.plot([xdict[mech,i+1,k],xdict[mech,i+1,k+1]],[ydict[mech,i+1,k],ydict[mech,i+1,k+1]],"-",color=couleur[i])
    plt.show()
    """
    solutionx=np.array([np.zeros(rotationIncrements),np.zeros(rotationIncrements),np.zeros(rotationIncrements),np.zeros(rotationIncrements),np.zeros(rotationIncrements),np.zeros(rotationIncrements),np.zeros(rotationIncrements),np.zeros(rotationIncrements)])
    solutiony=np.array([np.zeros(rotationIncrements),np.zeros(rotationIncrements),np.zeros(rotationIncrements),np.zeros(rotationIncrements),np.zeros(rotationIncrements),np.zeros(rotationIncrements),np.zeros(rotationIncrements),np.zeros(rotationIncrements)])
    
    
    
    for mech in range(nmc):
        if mech < 4 :
            for k in range(rotationIncrements):
                #plt.plot([xdict[mech,joint1,k],xdict[mech,joint1,k+1]],[ydict[mech,joint1,k],ydict[mech,joint1,k+1]],"-",color='blue')
                solutionx[mech][k]=xdict[mech,8,k]
                solutiony[mech][k]=ydict[mech,8,k]
        else: 
            for k in range(rotationIncrements):
                #plt.plot([xdict[mech,joint1,k],xdict[mech,joint1,k+1]],[ydict[mech,joint1,k],ydict[mech,joint1,k+1]],"-",color='blue')
                solutionx[mech][k]=xdict[mech,108,k]
                solutiony[mech][k]=ydict[mech,108,k]
    
    #plt.show()
    return solutionx,solutiony  

##############################################################################################################################################################    
# @jit(target ="cuda")
def f_inst_t(f,index):
    
    f_instant_t = []
    for i in range(len(f)):
        f_instant_t.append(f[i][index])  
    return f_instant_t
# @jit(target ="cuda")
def X_inst_t():
    Largeur_carter = 150
    Coef = 0.66
    X_inst_t =[Largeur_carter,Largeur_carter*Coef,Largeur_carter*(1-Coef),0,Largeur_carter,Largeur_carter*Coef,Largeur_carter*(1-Coef),0]
    return X_inst_t

##############################################################################################################################################################    
# @jit(target ="cuda")
def jambe_au_sol(L3p8,Z_jambe_instant_t,Y_jambe_instant_t,X_jambe_instant_t):
    """
    Detection des jambes qui touchent effectivement le sol 
    
    Parameters
    ----------
    L3p8 : Liste d'entiers
        Liste des 3 parmi 8 avec certaines valeurs ininterressantes retirées
    Z_jambe_instant_t : Liste de réels
        Coordonnées Z du pied à l'instant t
    Y_jambe_instant_t : Liste de réels
        Coordonnées Y du pied à l'instant t
    X_jambe_instant_t : Liste de réels
        Coordonnées X du pied à l'instant t

    Returns
    -------
    L3p8[i-1] : Tuple
    Triplet d'indices des Jambes touchants le sol 
    
    i-1 : Tuple
    Iindices dans le liste L3p8
    
    normale : Vecteur numpy
        Vecteur normal du plan formé par les 3 pieds touchants le sol
        
    d : Réel
        Constante du plan formé par les 3 pieds touchants le sol pour déterminer son équation cartesienne

    """
    Jambe_en_dessous = True
    i = 0
    Epsilon = 0.001
    while Jambe_en_dessous == True and i <len(L3p8):
        
        #print(L3p8[i][0]-1,L3p8[i][1]-1,L3p8[i][2]-1)
        A = [X_jambe_instant_t[L3p8[i][0]-1],Y_jambe_instant_t[L3p8[i][0]-1],Z_jambe_instant_t[L3p8[i][0]-1]]
        B = [X_jambe_instant_t[L3p8[i][1]-1],Y_jambe_instant_t[L3p8[i][1]-1],Z_jambe_instant_t[L3p8[i][1]-1]]
        C = [X_jambe_instant_t[L3p8[i][2]-1],Y_jambe_instant_t[L3p8[i][2]-1],Z_jambe_instant_t[L3p8[i][2]-1]]
        
        #print(A)
        vect_A = np.array(A)
        vect_B = np.array(B)
        vect_C = np.array(C)
        
        AB = vect_A-vect_B
        BC = vect_B-vect_C
                 
        normale = np.cross(AB, BC)
        d = - normale[0]*vect_A[0] - normale[1]*vect_A[1] - normale[2]*vect_A[2]
        #print(vect_A,vect_B,vect_C,AB,BC,normale,d)
        #print(normale)
        #print(A,B,C)
        Jambe_en_dessous = False 
        for jambe in range(1,len(X_jambe_instant_t)+1):
            
            if jambe not in L3p8[i] : 

                Z_plan =  (- normale[0]*X_jambe_instant_t[jambe-1] - normale[1]*Y_jambe_instant_t[jambe-1] - d) / normale[2]

                if Z_plan > Z_jambe_instant_t[jambe-1]+Epsilon:
                    Jambe_en_dessous = True
                    
        
        i = i + 1
        
    if Jambe_en_dessous == False : 
        return L3p8[i-1],i-1,normale,d
    else : 
        return None
            

##############################################################################################################################################################    
    
# @jit(target ="cuda")
def Delta_h(Liste_sol_t,L,l):
    """
    Calcul des hauteurs de Tangage et de roulis 
    
    Parameters
    ----------
    Liste_sol_t : Liste d'entiers
        Liste des jambes touchant le sol
    L : Entier
        Longueur du châssi du robot
    l : Entier
        Largeur du châssi du robot

    Returns
    -------
    Hauteur 1 : Réel
    Hauteur de tangage du robot
    
    Hauteur 2 : Réel
    Hauteur de Roulis du robot

    """
    x = np.array([1,0,0])
    y = np.array([0,1,0])
    n = Liste_sol_t[2]
    
    #nx = np.array([n[0],0,n[2]/2])
    #ny = np.array([0,n[1],n[2]/2])
    #print(nx,ny)
    #angle_x = np.arcsin( np.linalg.norm(np.cross(nx,z))/(np.linalg.norm(nx)+np.linalg.norm(z)))/np.pi*180-90
    #angle_y = np.arcsin( np.linalg.norm(np.cross(ny,z))/(np.linalg.norm(ny)+np.linalg.norm(z)))/np.pi*180-90
    
    angle_x = np.arccos( np.dot(n,x) / np.linalg.norm(n) )*180/np.pi-90
    angle_y = np.arccos( np.dot(n,y) / np.linalg.norm(n) )*180/np.pi-90

    return L * np.sin(angle_x * np.pi / 180), l * np.sin(angle_y * np.pi / 180)

###############################################################################
#DETECTION D'INTERFERENCES
###############################################################################
# @jit(target ="cuda")
def inter_segment(A,B,C,D):
    """
    Algorithme de detection d'intersection entre 2 segments
    
    Parameters
    ----------
    A,B : 2 points sous forme de vecteur numpy
    Deux points qui forment le 1er vecteur
    
    C,D : 2 points sous forme de vecteur numpy
    Deux points qui forment le 2ème vecteur

    Returns
    -------
    
    Retourne True si les deux segments se croisent et False sinon

    """
    if min(A[0],B[0]) > max(C[0],D[0]) or \
    min(C[0],D[0]) > max(A[0],B[0]) or \
    min(A[1],B[1]) > max(C[1],D[1]) or \
    min(C[1],D[1]) > max(A[1],B[1]) :
        return False
        
    AB = B-A
    CD = D-C
    
    alpha1 = AB[1]
    beta1 = -AB[0]
    gamma1 = -alpha1*A[0] - beta1*A[1]
    alpha2 = CD[1]
    beta2 = -CD[0]
    gamma2 = -alpha2*C[0] - beta2*C[1]
    
    x_inter = (gamma1*beta2-gamma2*beta1)/(beta1*alpha2-beta2*alpha1)
    
    res = False
    
    if x_inter <= min([max(A[0],B[0]),max(C[0],D[0])]) \
    and x_inter >= max([min(A[0],B[0]),min(C[0],D[0])]):
        res = True
    
    return res

###############################################################################
#FONCTION DE SIMULATION GLOBALE
###############################################################################
# @jit(target ="cuda")
def Simulation(nominal, ecartype, L3p8, RotationIncrements):
    """
    Simulation global du système Jansen avec toutes les fonctions précédentes
    
    Parameters
    ----------
    nominal : Liste de réels
        Longueurs nominales théoriques des barres qui composent la jambe 
    ecartype : Liste de réels
        Ecart Type pour la variation de longueur pour chaque barre
    nmc : Entier
        Nombre de système simulés
    rotationIncrements : Entier
        Nombre d'incrémentation pour la simulation
        Plus ce nombre est grand plus la simulation sera précise mais le temps de calcul sera augmenté

    Returns
    -------
    Z : Liste de réels
        Coordonnées Z de chaques jambes
        
    Y : Liste de réels
        Coordonnées Z de chaques jambes
        
    D : Liste de liste de réels
        Liste regroupant : Hauteur de tangage du robot et Hauteur de Roulis du robot

    """
    
    #Pamètres inérants à la simualation 
    Nmc = 8
    Longueur_carter = 100
    Largeur_carter = 150
    Liste_phi = [0,180,180,0,180,0,0,180]
    

    Y_jambe=[[],[],[],[],[],[],[],[]]
    Z_jambe=[[],[],[],[],[],[],[],[]]
    
    Sol =  Mecanisme_jambe (nominal, Ecartype, Nmc, RotationIncrements)
    
    for i in range(8):
        Y_jambe[i] = Sol [0][i]
        Z_jambe[i] = Sol [1][i]
    
    for l in range(len(Y_jambe)):
        if l < 4 :
            for m in range(len(Y_jambe[l])) :
                Y_jambe[l][m] = Y_jambe[l][m] - Longueur_carter/2
        elif l >= 4 :
            for m in range(len(Y_jambe[l])) :
                Y_jambe[l][m] = Y_jambe[l][m] + Longueur_carter/2
                
    Z = []
    Y = []
    
    for l in range(len(Z_jambe)):
        dephasage = len(Z_jambe[l])/360
        if dephasage*Liste_phi[l] != 0:
            Z.append(np.concatenate((Z_jambe[l][int(dephasage*Liste_phi[l]):-1], Z_jambe[l][:int(dephasage*Liste_phi[l])])))
            Y.append(np.concatenate((Y_jambe[l][int(dephasage*Liste_phi[l]):-1], Y_jambe[l][:int(dephasage*Liste_phi[l])])))
        else :
            Z.append(Z_jambe[l][:-1])
            Y.append(Y_jambe[l][:-1])

    Liste_pieds_au_sol = []
    for i in range(RotationIncrements-1):
    
        Liste_pieds_au_sol.append(jambe_au_sol(L3p8,f_inst_t(Z,i),f_inst_t(Y,i),X_inst_t()))
    
    
    
    D=[]
    for i in range(len(Liste_pieds_au_sol)):
        D.append(Delta_h(Liste_pieds_au_sol[i],Longueur_carter,Largeur_carter))
    
    D_x=[0]*len(D)
    D_y=[0]*len(D)

    for i in range(len(D)):
        D_x[i] = D[i][0]
        D_y[i] = D[i][1]
    
    Scorex = max(D_x)
    Scorey = max(D_y)
    
    return [Scorex, Scorey]


###############################################################################
#PARAMETRAGE ET LANCEMENT DE LA SIMULATION
###############################################################################

#Ecartype = np.array([0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02])
Sigma = 0.02
Ecartype = np.ones(13)*Sigma
RotationIncrements = 120


fichier = open("3 parmis 8.txt","r")
donnee = fichier.readlines()
fichier.close()


L3p8 = []
for k in range (len(donnee)):
    donnee[k] = donnee[k].rstrip("\n")
    donnee[k] = donnee[k].split(",")
    for l in range(len(donnee[k])):
        donnee[k][l] = int(donnee[k][l])
    L3p8.append(donnee[k])

#Simul = Simulation(Nominal, Ecartype, L3p8, RotationIncrements)

#Z_j = Simul[0]
#Y_j = Simul[1]
#D_h = Simul[2]

Simul = lambda L : Simulation(L, Ecartype, L3p8, RotationIncrements)

###############################################################################
#OPTIMISATION
###############################################################################

Nominal = np.array([1.5,3.8,4.15,3.97,4.01,5.58,3.94,3.67,6.57,4.9,5,6.19,0.78])

Sim = Simul(Nominal)

