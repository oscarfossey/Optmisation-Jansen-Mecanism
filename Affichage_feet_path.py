
import numpy as np
import matplotlib.pyplot as plt
import math
import openturns as ot



def Mecanisme_1_jambe (nominal, ecartype, nmc, rotationIncrements):

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
    
    joint1 = 8
    
    """
    couleur=['w','b', 'g', 'r', 'b', 'r', 'y', 'k']
    
    fig = plt.figure()  
    
    for mech in range(nmc):
        for i in range(8):
            for k in range(rotationIncrements-1):
                plt.plot([xdict[mech,i+1,k],xdict[mech,i+1,k+1]],[ydict[mech,i+1,k],ydict[mech,i+1,k+1]],"-",color=couleur[i])
    plt.show()
    """

    solutionx = np.zeros([nmc,rotationIncrements])
    solutiony = np.zeros([nmc,rotationIncrements])
    
    
    for mech in range(Nmc):
        for k in range(RotationIncrements):
            #plt.plot([xdict[mech,joint1,k],xdict[mech,joint1,k+1]],[ydict[mech,joint1,k],ydict[mech,joint1,k+1]],"-",color='blue')
            solutionx[mech][k]=xdict[mech,joint1,k]
            solutiony[mech][k]=ydict[mech,joint1,k]
    
    #plt.show()
    return solutionx,solutiony  

##############################################################################################################################################################
#                                       AFFICHAGE
##############################################################################################################################################################


#Longueurs nominales de fabrication
Nominal = np.array([1.4394682554527707, 3.990882217777572, 3.9724117441977977, 3.5671800757333543, 3.7377056662341723, 5.2942154572671525, 4.361415086048287, 3.901018088519162, 6.197087696022331, 4.710612274718331, 4.8311215810079196, 5.860699489693119, 0.7808132569821478])

#Ecart type dû au procédé de fabrication
Sigma = 0.02
Ecartype = np.ones(13)*Sigma

#Nombre de simulation
Nmc = 100


RotationIncrements = 120

Solx,Soly = Mecanisme_1_jambe (Nominal, Ecartype, Nmc, RotationIncrements)

for mech in range(Nmc):
    Len = int(len(Solx[mech-1]))
    plt.plot(Solx[mech-1][:Len],Soly[mech-1][:Len])
plt.show()




