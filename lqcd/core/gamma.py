#!/public/home/yanhb/software/miniconda3/bin/python3

# Return the value
from backend import get_backend
import os

# 2023.08.26: xp version.
# 2023.06.01: Returns the full gamma matrix.
# 2023.03.10: I inversed g1 and g3 to make it compatible with Chroma.

xp = get_backend()

#identity
g0=xp.zeros((4,4),dtype=complex)
g0[0,0]=1.0+0.0*1j
g0[1,1]=1.0+0.0*1j
g0[2,2]=1.0+0.0*1j
g0[3,3]=1.0+0.0*1j

#gamma1
#g1=xp.zeros((4,4),dtype=complex)
#g1[0,3]=0.0-1.0*1j
#g1[1,2]=0.0-1.0*1j
#g1[2,1]=0.0+1.0*1j
#g1[3,0]=0.0+1.0*1j

g1=xp.zeros((4,4),dtype=complex)
g1[0,3]=0.0+1.0*1j
g1[1,2]=0.0+1.0*1j
g1[2,1]=0.0-1.0*1j
g1[3,0]=0.0-1.0*1j

#gamma2
g2=xp.zeros((4,4),dtype=complex)
g2[0,3]=-1.0+0.0*1j
g2[1,2]=1.0+0.0*1j
g2[2,1]=1.0+0.0*1j
g2[3,0]=-1.0+0.0*1j

#gamma3
#g3=xp.zeros((4,4),dtype=complex)
#g3[0,2]=0.0-1.0*1j
#g3[1,3]=0.0+1.0*1j
#g3[2,0]=0.0+1.0*1j
#g3[3,1]=0.0-1.0*1j

g3=xp.zeros((4,4),dtype=complex)
g3[0,2]=0.0+1.0*1j
g3[1,3]=0.0-1.0*1j
g3[2,0]=0.0-1.0*1j
g3[3,1]=0.0+1.0*1j

#gamma4
g4=xp.zeros((4,4),dtype=complex)
g4[0,2]=1.0+0.0*1j
g4[1,3]=1.0+0.0*1j
g4[2,0]=1.0+0.0*1j
g4[3,1]=1.0+0.0*1j

#gamma5
g5=xp.zeros((4,4),dtype=complex)
g5[0,0]=1.0+0.0*1j
g5[1,1]=1.0+0.0*1j
g5[2,2]=-1.0+0.0*1j
g5[3,3]=-1.0+0.0*1j

def gamma(i):
    if i==0: #identity
        g=g0
        
    elif i==1: #gamma1
        g=g1
        
    elif i==2: #gamma2
        g=g2

    elif i==3: #gamma3
        g=g3
    
    elif i==4: #gamma4
        g=g4
    
    elif i==5: #gamma5
        g=g5

    elif i==6: #-gamma1*gamma4*gamma5 (gamma2*gamma3)
        g=xp.matmul(g2,g3)
        
    elif i==7: #-gamma2*gamma4*gamma5 (gamma3*gamma1)
        g=xp.matmul(g3,g1)
 
    elif i==8: #-gamma3*gamma4*gamma5 (gamma1*gamma2)
        g=xp.matmul(g1,g2)
 
    elif i==9: #gamma1*gamma4
        g=xp.matmul(g1,g4)
 
    elif i==10: #gamma2*gamma4
        g=xp.matmul(g2,g4)
 
    elif i==11: #gamma3*gamma4
        g=xp.matmul(g3,g4)
 
    elif i==12: #gamma1*gamma5
        g=xp.matmul(g1,g5)
 
    elif i==13: #gamma2*gamma5
        g=xp.matmul(g2,g5)
 
    elif i==14: #gamma3*gamma5
        g=xp.matmul(g3,g5)
 
    elif i==15: #gamma4*gamma5
        g=xp.matmul(g4,g5)

    else:
        print("wrong gamma index")
        os.sys.exit(-3)

    return g


def gamma_index(i):
    if i==0: #identity
        g=g0
        
    elif i==1: #gamma1
        g=g1
        
    elif i==2: #gamma2
        g=g2

    elif i==3: #gamma3
        g=g3
    
    elif i==4: #gamma4
        g=g4
    
    elif i==5: #gamma5
        g=g5

    elif i==6: #-gamma1*gamma4*gamma5 (gamma2*gamma3)
        g=xp.matmul(g2,g3)
        
    elif i==7: #-gamma2*gamma4*gamma5 (gamma3*gamma1)
        g=xp.matmul(g3,g1)
 
    elif i==8: #-gamma3*gamma4*gamma5 (gamma1*gamma2)
        g=xp.matmul(g1,g2)
 
    elif i==9: #gamma1*gamma4
        g=xp.matmul(g1,g4)
 
    elif i==10: #gamma2*gamma4
        g=xp.matmul(g2,g4)
 
    elif i==11: #gamma3*gamma4
        g=xp.matmul(g3,g4)
 
    elif i==12: #gamma1*gamma5
        g=xp.matmul(g1,g5)
 
    elif i==13: #gamma2*gamma5
        g=xp.matmul(g2,g5)
 
    elif i==14: #gamma3*gamma5
        g=xp.matmul(g3,g5)
 
    elif i==15: #gamma4*gamma5
        g=xp.matmul(g4,g5)

    else:
        print("wrong gamma index")
        os.sys.exit(-3)

    value=[]
    row=[]
    col=[]
    for i in range(4):
        for j in range(4):
            if(xp.abs(g[i,j]) != 0.0):
                value.append(g[i,j])        
                row.append(i)
                col.append(j)
    return value, row, col 

def gammamul_index(i,j):
    if i==0: #identity
        gi=g0
        
    elif i==1: #gamma1
        gi=g1
        
    elif i==2: #gamma2
        gi=g2

    elif i==3: #gamma3
        gi=g3
    
    elif i==4: #gamma4
        gi=g4
    
    elif i==5: #gamma5
        gi=g5

    elif i==6: #-gamma1*gamma4*gamma5 (gamma2*gamma3)
        gi=xp.matmul(g2,g3)
        
    elif i==7: #-gamma2*gamma4*gamma5 (gamma3*gamma1)
        gi=xp.matmul(g3,g1)
 
    elif i==8: #-gamma3*gamma4*gamma5 (gamma1*gamma2)
        gi=xp.matmul(g1,g2)
 
    elif i==9: #gamma1*gamma4
        gi=xp.matmul(g1,g4)
 
    elif i==10: #gamma2*gamma4
        gi=xp.matmul(g2,g4)
 
    elif i==11: #gamma3*gamma4
        gi=xp.matmul(g3,g4)
 
    elif i==12: #gamma1*gamma5
        gi=xp.matmul(g1,g5)
 
    elif i==13: #gamma2*gamma5
        gi=xp.matmul(g2,g5)
 
    elif i==14: #gamma3*gamma5
        gi=xp.matmul(g3,g5)
 
    elif i==15: #gamma4*gamma5
        gi=xp.matmul(g4,g5)

    else:
        print("wrong gamma index")
        os.sys.exit(-3)

    if j==0: #identity
        gj=g0
        
    elif j==1: #gamma1
        gj=g1
        
    elif j==2: #gamma2
        gj=g2

    elif j==3: #gamma3
        gj=g3
    
    elif j==4: #gamma4
        gj=g4
    
    elif j==5: #gamma5
        gj=g5

    elif j==6: #-gamma1*gamma4*gamma5 (gamma2*gamma3)
        gj=xp.matmul(g2,g3)
        
    elif j==7: #-gamma2*gamma4*gamma5 (gamma3*gamma1)
        gj=xp.matmul(g3,g1)
 
    elif j==8: #-gamma3*gamma4*gamma5 (gamma1*gamma2)
        gj=xp.matmul(g1,g2)
 
    elif j==9: #gamma1*gamma4
        gj=xp.matmul(g1,g4)
 
    elif j==10: #gamma2*gamma4
        gj=xp.matmul(g2,g4)
 
    elif j==11: #gamma3*gamma4
        gj=xp.matmul(g3,g4)
 
    elif j==12: #gamma1*gamma5
        gj=xp.matmul(g1,g5)
 
    elif j==13: #gamma2*gamma5
        gj=xp.matmul(g2,g5)
 
    elif j==14: #gamma3*gamma5
        gj=xp.matmul(g3,g5)
 
    elif j==15: #gamma4*gamma5
        gj=xp.matmul(g4,g5)

    else:
        print("wrong gamma index")
        os.sys.exit(-3)

    g=xp.matmul(gi,gj)
    
    value=[]
    row=[]
    col=[]
    for i in range(4):
        for j in range(4):
            if(xp.abs(g[i,j]) != 0.0):
                value.append(g[i,j])        
                row.append(i)
                col.append(j)
    return value, row, col 
