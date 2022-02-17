# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:36:04 2020

@author: admin
"""

import statistics
import math
import numpy as np
import matplotlib.pyplot as plt
import operator
# from numba import jit, cuda

ntest_par_generation = 30#nombre d'individu par generation
n_generation = 25

quantite_selection = 40
quantite_croissement = 10
quantite_mutation = 20
taux_mutation = 0.1

def f(L):
    """ça c'est une fonction test a remplacer par la fonction qui t'interesse L = liste avec les variables le retour sous forme de liste également ici un couple [,]"""
    
    x = L[0] 
    y = L[1] 
    z = L[2] 
    
    return [(x-3)**2 + (y-2)**2 + (z+7)**2,0]

def score(function, parametre, objectif):
    
    y = [i for i in function(parametre)]
    

    return 1/(math.sqrt(sum([(y[i]-objectif[i])**2 for i in range(len(objectif))]))+1)

# @jit(target ="cuda")
def evolution(generation, intervalles, objectif, function):
    """Cette fonction s'occupe de faire évoluer UNE generation on prend les iindividus de la generations père avec le meilleure score 
    ensuite avec des croissement et mutation et injection (juste avec des valeurs aléatoire dans l'interval) je creer la generation fils """
    
    # Scores = {}
                                                       
    new_generation = [{},{},generation[2]+1]
    
    # score_limite = np.percentile(list(generation[1].values()), 100-quantite_selection)
    
    gen_trier = sorted(generation[1].items(), key=operator.itemgetter(1), reverse = True)
    
    Selectiones = [generation[0][test[0]] for test in gen_trier[:ntest_par_generation*quantite_selection//100]]
    # Selectiones = [parametre for parametre in list(generation[0].values()) if list(generation[1].values())[list(generation[0].values()).index(parametre)] >= score_limite] #selection
    
    k = 0
    
    for test in gen_trier[:ntest_par_generation*quantite_selection//100]:
        
        new_generation[0]['g' + str(generation[2]+1) + 't' + str(k)] = generation[0][test[0]]
        new_generation[1]['g' + str(generation[2]+1) + 't' + str(k)] = test[1]
        k += 1
    
    j = 0
    
    while j < quantite_mutation*ntest_par_generation//100 and k < ntest_par_generation :
        
        parametre_ref = Selectiones[j]
        
        new_parametre = []
        
        for val_ref in parametre_ref:
            
            val_new = val_ref*(1+2*taux_mutation*(np.random.random()-0.5))                                 #mutation
            
            I = intervalles[parametre_ref.index(val_ref)]
            
            if val_new > I[0] and val_new < I[1]:
                
                new_parametre.append(val_new)
            
            else:
                
                new_parametre.append(val_ref)
        
        new_generation[0]['g' + str(generation[2]+1) + 't' + str(k)] = new_parametre
        new_generation[1]['g' + str(generation[2]+1) + 't' + str(k)] = score(function, new_parametre, objectif)        
        
        k += 1
        j += 1
    
    l = 0
    
    while l < ntest_par_generation*quantite_croissement//100 and k < ntest_par_generation:
        
        i=np.random.randint(len(Selectiones))
        j=np.random.randint(len(Selectiones))
        para1 = Selectiones[i]
        para2 = Selectiones[j]
        
        new_parametre = []
        
        for i in range(len(para1)):
            
            val_new = (para1[i]+para2[i])/2                                                  # croissement
            
            I = intervalles[i]
            
            if val_new > I[0] and val_new < I[1]:
                
                new_parametre.append(val_new)
            
            else:
                
                para = np.random.choice([para1, para2])
                new_parametre.append(para)
        
        new_generation[0]['g' + str(generation[2]+1) + 't' + str(k)] = new_parametre                      
        new_generation[1]['g' + str(generation[2]+1) + 't' + str(k)] = score(function, new_parametre, objectif)   
        
        k += 1
        l += 1
    
    while k < ntest_par_generation:
        
        new_parametre = []
        
        for I in intervalles:
        
            new_parametre += [I[0]+np.random.random_sample()*(I[1]-I[0])]                            #injection random
        
        new_generation[0]['g' + str(generation[2]+1) + 't' + str(k)] = new_parametre                        
        new_generation[1]['g' + str(generation[2]+1) + 't' + str(k)] = score(function, new_parametre, objectif)   
        
        k += 1
    
    return new_generation


# @jit(target ="cuda")
def minimize(intervalles, function, objectif):
    """Cette fonction est la fonction principal qu'il faut utiliser
    intervalles sous forme de liste par exemple [[0,10],[-5,5]] ici la variable x va de 0 à 10 et la variable y de -5 à 5
    function : cf la forme de f (on travail encore avec des listes)
    objectif : doit avoir la meme forme que la sortie de f (donc cf f)"""
    
    
    generation0 = [{}, {}, 0]
    
    for test in range(ntest_par_generation):
        
        parametre = []
        # parametre = [1.2122037465344935, 3.8761459377458705, 4.203715339548159, 4.030068719260273, 3.7841850129145915, 5.607961882924281, 3.8986131848949457, 3.7876063490756815, 6.566217272352248, 5.059916327591068, 4.9260458745399065, 6.100684596526857, 0.8451359799458538]
    
        for I in intervalles:
        
            parametre += [I[0]+np.random.random_sample()*(I[1]-I[0])]
        
        generation0[0]['g0t'+ str(test)] = parametre
        generation0[1]['g0t'+str(test)] = score(function, parametre, objectif)
    
    generation = [generation0]
    
    for n in range(1,n_generation):
        
        print(str(100*n//n_generation) + " %")
        
        generation += [evolution(generation[-1], intervalles, objectif, function)]
    
    """Maintenant que nous avons tout l'évolution on prend les informations qui nous interresse (moyenne,Q1,Q3 et surtout le max de chaque gen)
    soit les donnes sur l'évolution total et évidement les variables qui nous améne le plus proche de l'objectif après ngen generation """
    
    num_generation =[]
    scores_max = []
    scores_Q1 = []
    scores_Q3 =[]
    scores_moyenne = []
    
    for g in generation:
        
        num_generation += [g[-1]]
        
        g_trier = sorted(g[1].items(), key=operator.itemgetter(1), reverse = True)
        

        parametre_max = g[0][g_trier[0][0]]
            
        S = list(g[1].values())
        
        scores_max += [S[0]]
        scores_Q1 += [np.percentile(S, 25)]
        scores_Q3 += [np.percentile(S, 75)]
        scores_moyenne += [statistics.mean(S)]
    
    
    plt.plot(num_generation,scores_max,'b', linewidth=3)    
    plt.plot(num_generation,scores_Q1,'b--',linewidth=1)   
    plt.plot(num_generation,scores_Q3,'b--',linewidth=1)   
    plt.plot(num_generation,scores_moyenne,'b',linewidth=2)
    
    print("score le plus élevé " + str(scores_max[-1]))
    
    return function(parametre_max),parametre_max
    
        
        
    
    