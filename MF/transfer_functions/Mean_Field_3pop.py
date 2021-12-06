def input_rate(t,t1_exc,tau1_exc,tau2_exc,ampl_exc):
    #t1_exc=10. # time of the maximum of external stimulation 
    #tau1_exc=20. # first time constant of perturbation = rising time 
    #tau2_exc=50. # decaying time 
    #ampl_exc=20. # amplitude of excitation
    inp=ampl_exc*(np.exp(-(t-t1_exc)**2/(2.*tau1_exc**2))*heaviside(-(t-t1_exc))+\
    np.exp(-(t-t1_exc)**2/(2.*tau2_exc**2))*heaviside(t-t1_exc))
    return inp

t = np.arange(0, 2, 0.0001)
test_input=[]


for i in t:
   # print(i), print (input_rate(i))
    test_input.append(input_rate(i,0.5,0.05,0.2,5))



######### INTEGRATION OF SECOND ORDER OF MF MODEL ###############

from scipy.integrate import odeint
from random import *

h=0.001
T=0.005


# First model obtained by splitting the excitatory pop in 2 equal pop => 

def MeanField_dynamics(state, t):
    ve, vi, cee, cii, cei, cie, vb, ceb, cib = state
    v_exc=(ve+vb)/2+10.+input_rate(t,2,0.1,0.3,5)
    d_ve = (-ve + TF1(v_exc, vi) + 0.5 * (second_deriv_11(TF1,v_exc,vi,h)*(cee+cbb+2*ceb)+cei*second_deriv_12(TF1,v_exc,vi,h)+cie*second_deriv_21(TF1,v_exc,vi,h)+cib*second_deriv_12(TF1,v_exc,vi,h)+cbi*second_deriv_21(TF1,v_exc,vi,h)+cii*second_deriv_22(TF1,v_exc,vi,h)))/T
    d_vi = (-vi + TF2(v_exc, vi)+ 0.5 * (second_deriv_11(TF2,v_exc,vi,h)*(cee+cbb+2*ceb)+cei*second_deriv_12(TF2,v_exc,vi,h)+cie*second_deriv_21(TF2,v_exc,vi,h)+cib*second_deriv_12(TF2,v_exc,vi,h)+cbi*second_deriv_21(TF2,v_exc,vi,h)+cii*second_deriv_22(TF2,v_exc,vi,h)))/T
    d_vb = (-vb + TF2(v_exc, vi)+ 0.5 * (second_deriv_11(TF1,v_exc,vi,h)*(cee+cbb+2*ceb)+cei*second_deriv_12(TF1,v_exc,vi,h)+cie*second_deriv_21(TF1,v_exc,vi,h)+cib*second_deriv_12(TF1,v_exc,vi,h)+cbi*second_deriv_21(TF1,v_exc,vi,h)+cii*second_deriv_22(TF1,v_exc,vi,h)))/T
    d_cee = ((TF1(v_exc,vi)*(1/T-TF1(v_exc,vi)))/8000 + (TF1(v_exc,vi)-ve)**2+2*(cee*discrete_deriv_1(TF1,v_exc,vi,h)+cei*discrete_deriv_2(TF1,v_exc,vi,h)) - 2*cee)/T
    d_cii = ((TF2(v_exc,vi)*(1/T-TF2(v_exc,vi)))/2000+ (TF2(v_exc,vi)-vi)**2+2*(cie*discrete_deriv_1(TF2,v_exc,vi,h)+cii*discrete_deriv_2(TF2,v_exc,vi,h)) - 2*cii)/T
    d_cei = ((TF1(v_exc,vi)-ve)*(TF2(v_exc,vi)-vi)+cie*discrete_deriv_1(TF1,v_exc,vi,h)+cii*discrete_deriv_2(TF1,v_exc,vi,h)+cee*discrete_deriv_1(TF2,v_exc,vi,h)+cei*discrete_deriv_2(TF2,v_exc,vi,h) - 2*cei)/T
    d_cie = ((TF1(v_exc,vi)-ve)*(TF2(v_exc,vi)-vi)+cie*discrete_deriv_1(TF1,v_exc,vi,h)+cii*discrete_deriv_2(TF1,v_exc,vi,h)+cee*discrete_deriv_1(TF2,v_exc,vi,h)+cei*discrete_deriv_2(TF2,v_exc,vi,h) - 2*cie)/T

    return [d_ve, d_vi,d_cee,d_cii,d_cei,d_cie]


###########################################################################
    
            ###### MOST GENERAL NOTATION ######
            
###########################################################################
            
def MeanField_general(x0, params, t):
    
    tauw=params[2]
    a=params[3]
    b=params[4]
    El=params[5]
    N=params[6]
    numb_var=params[7]
    numb_adapt=params[8]
    
    v=[]
    c=[]
    w=[]
    
    index=0
    
    ''' Reshape the variables for more clarity '''
    
    for i in range(numb_var):
        v.append(x0[i])
        c_temp=[]
        for j in range(numb_var):
            c_temp.append(x0[index+numb_var])
            index+=1
        c.append(c_temp)
        
    index=0 # rester index   
    
    for k in range(numb_adapt):
        w.append(x0[numb_var*numb_var+numb_var+index])
        index+=1
    
    
    A = A_function(N,v,T,numb_var)
        
    v_exc=(v[0]+v[1])/2+10.+input_rate(t,2,0.1,0.3,5)
    
    for i in range(numb_var):
        dv[i]=(F[i](v)-v[i])
        for j in range(numb_var):
            for k in range(numb_var): 
                dv[i]+=0.5*deriv2[i][k](F[i](v))*c[j][k] # if i and k =! 2 deriv is second deriv
        dc[i][j]=np.kron(i,j)*A[i]+(F[i](v)-v[i])*(F[j](v)-v[j])
        
    for k in range(numb_var):
        dc[i][j]+=c[j][k]*deriv1[k](F[i](v))+c[i][k]*deriv1[k](F(j,v))-2*c[i][j]
    for i in range(numb_adapt):
        dw[i]=-w[i]/tauw[i]+b[i]*v[i]+a[i]*(mu[i](v,w)-El[i])
        
        
    ''' Reshape the deriv to plug it into RK4 routine '''
        
    deriv=[]
    
    for i in range(numb_var):
        deriv.append(v[i])
    
    for i in range(numb_var):
        for j in range(numb_var):
            deriv.append(c[i][j])
    
    for i in range(numb_adapt):
        deriv.append(dw[i])
        
        
    return deriv
        
    
def Transfer_functions(v,numb_var):
    
    TF = load_transfer_functions(NRN1, NRN2, NTWK)
    F = ()
    for i in range(numb_var):
        F = F + (TF[i](v[0],v[1]),)
        F = F + (TF[i](v[0],v[1]),)        
    
    return F

def A_function(N,v,T,numb_var): # N is an array with the numbers of elements in each pop
    
    TF = load_transfer_functions(NRN1, NRN2, NTWK)
    A=()
    for i in range(numb_var):
        A = A+(N[i]/(TF[i](v)*(1/T-TF[i])))
    
def first_deriv(funct,v,numb_var,h):
    deriv1=()
    vsec_plus=np.ones(numb_var)
    vsec_minus=np.ones(numb_var)
    
    for i in range(numb_var): # define perturbation vector
        vsec_plus[i]=v[i]
        vsec_minus[i]=v[i]
    for i in range(numb_var):
        vsec_plus[i]=v[i]+0.5*h # perturb one component
        vsec_minus[i]=v[i]-0.5*h
        deriv1 = deriv1 + ((funct(vsec_plus)-funct(vsec_minus))/h,)
        vsec_plus[i]=v[i] # reset component
        vsec_minus[i]=v[i]
        
    return deriv1

def sec_deriv(funct,v,numb_var,h):  
    # separate first and second deriv to avoid useless confusion
    deriv2=()
    vsec_plus=np.ones(numb_var)
    vsec_minus=np.ones(numb_var)
    deriv1=first_deriv(funct,v,numb_var,h)
    
    for i in range(num_var):
        deriv_2_temp=()
        for j in range(numb_var):
            vsec_plus[j]=v[j]+0.5*h # perturb one component
            vsec_minus[j]=v[j]-0.5*h    
            deriv_2_temp = deriv_2_temp + ((deriv1[i](funct,vsec_plus,h)-deriv1[i](funct,vsec_minus,vi,h))/h,)
            vsec_plus[j]=v[j] # reset component
            vsec_minus[j]=v[j]
        deriv2=deriv2+(deriv_2_temp,)
        
     return deriv2
        
        
            
            
# simulate now dynamics
#t = np.arange(0, 100, 0.0001)
init_state = [1.,1.,1.,1., 1.,1.]
state = odeint(MeanField_dynamics, init_state, t)


# Include bursters but keep 1 TF


def MeanField_dynamics(state, t):
    ve, vi, cee, cii, cei, cie, vb, ceb, cib = state
    v_exc=ve+10.+input_rate(t,2,0.1,0.3,5)
    d_ve = (-ve + TF1(ve_sec+vb, vi) + 0.5 * (cee*second_deriv_11(TF1,ve_sec+vb,vi,h)+cei*second_deriv_12(TF1,ve_sec+vb,vi,h)+cie*second_deriv_21(TF1,ve_sec+vb,vi,h)++cii*second_deriv_22(TF1,ve_sec+vb,vi,h)))/T
    d_vi = (-vi + TF2(ve_sec+vb, vi)+ 0.5 * (cee*second_deriv_11(TF2,ve_sec+vb,vi,h)+cei*second_deriv_12(TF2,ve_sec+vb,vi,h)+cie*second_deriv_21(TF2,ve_sec+vb,vi,h)+cii*second_deriv_22(TF2,ve_sec+vb,vi,h)))/T
    d_vb = (-vb + TF3(ve_sec+vb, vi)+ 0.5 * (cee*second_deriv_11(TF2,ve_sec+vb,vi,h)+cei*second_deriv_12(TF2,ve_sec+vb,vi,h)+cie*second_deriv_21(TF2,ve_sec+vb,vi,h)+cii*second_deriv_22(TF2,ve_sec+vb,vi,h)))/T
    d_cee = ((TF1(ve_sec+vb,vi)*(1/T-TF1(ve_sec+vb,vi)))/8000 + (TF1(ve_sec+vb,vi)-ve)**2+2*(cee*discrete_deriv_1(TF1,ve_sec+vb,vi,h)+cei*discrete_deriv_2(TF1,ve_sec+vb,vi,h)) - 2*cee)/T
    d_cii = ((TF2(ve_sec+vb,vi)*(1/T-TF2(ve_sec+vb,vi)))/2000+ (TF2(ve_sec+vb,vi)-vi)**2+2*(cie*discrete_deriv_1(TF2,ve_sec+vb,vi,h)+cii*discrete_deriv_2(TF2,ve_sec+vb,vi,h)) - 2*cii)/T
    d_cei = ((TF1(ve_sec+vb,vi)-ve)*(TF2(ve_sec+vb,vi)-vi)+cie*discrete_deriv_1(TF1,ve_sec+vb,vi,h)+cii*discrete_deriv_2(TF1,ve_sec+vb,vi,h)+cee*discrete_deriv_1(TF2,ve_sec+vb,vi,h)+cei*discrete_deriv_2(TF2,ve_sec+vb,vi,h) - 2*cei)/T
    d_cie = ((TF1(ve_sec+vb,vi)-ve)*(TF2(ve_sec+vb,vi)-vi)+cie*discrete_deriv_1(TF1,ve_sec+vb,vi,h)+cii*discrete_deriv_2(TF1,ve_sec+vb,vi,h)+cee*discrete_deriv_1(TF2,ve_sec+vb,vi,h)+cei*discrete_deriv_2(TF2,ve_sec+vb,vi,h) - 2*cie)/T

    return [d_ve, d_vi,d_cee,d_cii,d_cei,d_cie]


####### DEFINE RUNGE KUTTA ALGO #############

def rk4(x0,g,par)

	p=par[1]# // variables needed to build arrays
	#int lp = par[8];
	dx=np.zeros(p)  
	deriv=np.zeros(p)
	xt = np.zeros(p)
#	double **gderiv, **gt, **dg;
#	dg= new double*[lp];
#	gderiv = new double*[lp];
#	gt= new double*[lp];
	
	

#	for (int i=0;i<lp;i++) 
#	{
#		dg[i]= new double[p];
#		gderiv[i]= new double[p];
#		gt[i]=new double[p];
#
#	}

	for i=0 in range(p):
		dx[i]=deriv[i]=xt[i]=0
#		for (int j=0;j<lp;j++)
#		{
#			gderiv[j][i]=0;
#			gt[j][i]=0;
#		}

	# 1st RK step

	#phasespace(x0,deriv,par); #k1 
	#tanspace(g,gderiv,x0,par); // g1
    
    deriv=MeanField_general(x0, params, t)

	for i=0 in range(p): 
		dx[i]=deriv[i] # x[i]=k1[i]
		xt[i]=x0[i]+0.5*deriv[i] # argument for k2 (careful k1,2,3,4 in comment = k's of RK method, different from k1 & k2 of the code)

#	for (int j=0;j<lp;j++) 
#	{
#		for (int i=0;i<par[3];i++) 
#		{
#			dg[j][i]=gderiv[j][i];
#			gt[j][i]=g[j][i]+0.5*gderiv[j][i];
#		}
#	}

	time+=par[0]/2;
	
	# 2nd RK step
	phasespace(xt,deriv,par) # k2[i]=f(x0[i] +tstep/2*k1[i])
#	tanspace(gt,gderiv,xt,par); # g2

	for i=0 in range(p):
		dx[i]+=2*deriv[i] # x[i]=k1[i]+2*k2[i]
		xt[i]=x0[i]+0.5*deriv[i] # argument for k3	


	
#	for (int j=0;j<lp;j++) 
#	{
#		for (int i=0;i<par[3];i++) 
#		{
#			dg[j][i]+=2*gderiv[j][i];
#			gt[j][i]=g[j][i]+gderiv[j][i]*0.5;
#		}
#	}
	
	# 3rd RK step
	phasespace(xt,deriv,par) # k3 = f(x0[i] + tstep*k2[i]/2)
	#tanspace(gt,gderiv,xt,par); // g3

	for i=0 in range(p):
		dx[i]+=2*deriv[i] 
		xt[i]=x0[i]+deriv[i] # //argument for k4


#	for (int j=0;j<lp;j++) 
#	{
#		for (int i=0;i<par[3];i++) 
#		{
#			dg[j][i]+=2*gderiv[j][i];
#			gt[j][i]=g[j][i]+gderiv[j][i];
#		}
#	}
	
	#time+=par[0]/2;
	
	# 4th step
	phasespace(xt,deriv,par) # //k4[i]= f(x0[i] + tstep*k3[i])
	#tanspace(gt,gderiv,xt,par); //g4

	for i=0 in range(p):
        dx[i]+=deriv[i] # x[i]=k1[i]+2*k2[i]
		xt[i]=x0[i]+deriv[i] 

	


	for i=0 in range(p):
        x0[i]=x0[i]+dx[i]/6.0
        
#		for (int j=0;j<lp;j++) 
#		{
#			dg[j][i]+=gderiv[j][i];
#			g[j][i]+=dg[j][i]/6.0;
#		}
	
	
	
