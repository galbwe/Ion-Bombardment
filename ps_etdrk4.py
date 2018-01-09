from __future__ import division
import numpy as np
from numpy.fft import rfft2,irfft2
import math
from numpy.random import rand, randn, seed
from scipy.fftpack import fftshift,fftfreq
from scipy.integrate import simps,trapz,quad
import shutil
import time
import sys
import pdb

#solves the system of evolution equations
#u_t = v - a*grad^2*u - grad^2grad^2*u + lamda*(grad u)^2
#v_t = -v + b*grad^2*u - c*grad^2*v - d*grad^2grad^2*v + nu*v^2 + eta*v^3 + zeta*grad^2*v*3

def solve_ps(N,Nfinal,h,ckeep,L,a,b,c,d,lamda,nu,eta,zeta,filename = None):
    
    print 'initializing ps_etdrk4'
    print 'N = %s' % N
    print 'Nfinal = %s' %Nfinal
    print 'h = %s' %h
    print 'ckeep = %s' %ckeep
    print 'L = %s' %L
    print 'a = %s' %a
    print 'b = %s' %b
    print 'c = %s' %c
    print 'd = %s' %d
    print 'lamda = %s' %lamda
    print 'nu = %s' %nu
    print 'eta = %s' %eta
    print 'zeta = %s' %zeta
    
    u,v = np.array((1e-3)*rand(N,N)), np.array((1e-3)*rand(N,N))#initial conditions
    uhat = rfft2(u)
    vhat = rfft2(v)    
    ukeep = np.empty((1 + int(Nfinal/ckeep),N,N))
    ukeep[0,:,:] = u
    vkeep = np.empty((1 + int(Nfinal/ckeep),N,N))
    vkeep[0,:,:] = v
    tkeep = h*np.arange(0,Nfinal+1,ckeep)
    ky = np.pi/L*fftshift(np.arange(-N/2,N/2))
    kx = np.pi/L*np.arange(N/2+1)
    kxx,kyy = np.meshgrid(kx,ky)
    ksq = kxx**2 + kyy**2   

    #Diagonal entries of the linear operators Lt and Lt/2 
    Lu = h*(a*ksq - ksq**2) 
    Lu2 = 0.5*Lu
    Lv = h*(-1 + c*ksq-d*ksq**2)
    Lv2 = 0.5*Lv

    M = 32#number of points to use in complex contour integral        
    t = np.linspace(0,np.pi,M)
    rts = np.exp(1j*t)
    #computes f(L) using contour integral over circle of radius 1 centered at L 
    def evaluate(f,L):
        integrand = f(L.reshape((L.shape[0],L.shape[1],1)) + rts)
        return np.real(np.trapz(integrand,dx=np.pi/(M-1)))/np.pi        
 
    coeff_start = time.time()

	#functions kappa, alpha, beta, and gamma are defined at bottom,
	# and are taken from Trefethen, Kassam, 2005
    kappa_u = evaluate(kappa,Lu2)
    kappa_v = evaluate(kappa,Lv2)
    alpha_u = evaluate(alpha,Lu)
    alpha_v = evaluate(alpha,Lv)
    beta_u = evaluate(beta,Lu)
    beta_v = evaluate(beta,Lv)
    gamma_u = evaluate(gamma,Lu)
    gamma_v = evaluate(gamma,Lv)

    coeff_end = time.time()
    coeff_time = coeff_end - coeff_start
    print 'Coefficients computed in %s seconds' % coeff_time   
     
    def periodic_gradient(u,dx):
        u_ext = np.zeros((u.shape[0]+4,u.shape[1]+4))
        u_ext[2:N+2,2:N+2] = u
        u_ext[2:N+2,N+2:] = u[:,0:2]
        u_ext[2:N+2,0:2] = u[:,N-2:]
        u_ext[N+2:,2:N+2] = u[0:2,:]
        u_ext[0:2,2:N+2] = u[N-2:,:]
        u_ext[N+2:,N+2:] = u[0:2,0:2]
        u_ext[0:2,N+2:] = u[N-2:,0:2]
        u_ext[N+2:,0:2] = u[0:2,N-2:]
        u_ext[0:2,0:2] = u[N-2:,N-2:]
        uy_ext,ux_ext = np.gradient(u_ext,dx)
        uy = uy_ext[2:N+2,2:N+2]
        ux = ux_ext[2:N+2,2:N+2]
        return uy,ux

    def rhs(u,v): #Nonlinear parts of the original pde        
        uy,ux = periodic_gradient(u,2*L/N)
        v3 = v**3
        v3y,v3x = periodic_gradient(v3,2*L/N)
        v3yy,v3yx = periodic_gradient(v3y,2*L/N)
        v3xy,v3xx = periodic_gradient(v3x,2*L/N)		
        return lamda*(ux**2 + uy**2), nu*v**2 + eta*v3 + zeta*(v3xx + v3yy)

    #nonlinear operator on fourier space
    #N_u(uhat,vhat) = fft(rhs_u(real(ifft(uhat,vhat)))) 
        #N_v(uhat,vhat) = fft(rhs_v(real(ifft(uhat,vhat)))) - b*ksq*uhat
    def nonlinear(uhat,vhat):
        u = irfft2(uhat)
        v = irfft2(vhat)
        u,v = rhs(u,v)
        return rfft2(u) + vhat,rfft2(v)-b*ksq*uhat    
    
    #diagonal entries of matrix exponential exp(Lt) 
    eu2 = np.exp(Lu2)
    eu = np.exp(Lu)
    ev2 = np.exp(Lv2)
    ev = np.exp(Lv)
    #etdrk4 time stepping loop

    loop_start = time.time()
    for n in range(1,Nfinal+1):       

        Nu,Nv = nonlinear(uhat,vhat)
        #the multiplicative factor of 0.5*h is to account for the discrepancy in
        #the Cox/Matthews ETDRK4 formula and our coefficients after we have evaluated kappa
        au = eu2*uhat + 0.5*h*kappa_u*Nu 
        av = ev2*vhat + 0.5*h*kappa_v*Nv

        Nau,Nav = nonlinear(au,av)
        bu = eu2*uhat + 0.5*h*kappa_u*Nau
        bv = ev2*vhat + 0.5*h*kappa_v*Nav

        Nbu,Nbv = nonlinear(bu,bv)
        cu = eu2*au + .5*h*kappa_u*(2*Nbu-Nu)
        cv = ev2*av + .5*h*kappa_v*(2*Nbv-Nv)

        Ncu,Ncv = nonlinear(cu,cv)

        uhat = rfft2(irfft2(eu*uhat + h*alpha_u*Nu + 2*h*beta_u*(Nau + Nbu) + h*gamma_u*Ncu))
        vhat = rfft2(irfft2(ev*vhat + h*alpha_v*Nv + 2*h*beta_v*(Nav + Nbv) + h*gamma_v*Ncv))

        try:
            if np.isnan(uhat).any() or np.isnan(vhat).any():
                raise ValueError('Solution Contains NaN') 
        except ValueError as error:
            print(error.args)
            break

        if int(n%ckeep) == 0:           

            ukeep[int(n/ckeep),:,:] = irfft2(uhat)
            vkeep[int(n/ckeep),:,:] = irfft2(vhat)     

    loop_end = time.time()
    loop_time = loop_end-loop_start
    print 'ETDRK4 loop completed in %s seconds' % loop_time

    if filename == None:
        filename = ('phase_sep_etdrk4_' + time.strftime('%m_%d_%Y_%H_%M') + '_a_' 
        	+ str(a) + '_b_' + str(b) + '_c_' + str(c) + '_d_' 
        	+ str(d) + '_L_' + str(L) + '_N_' + str(N) + '_h_' + str(h))
        filename = filename.replace('.','_').replace('-','m')
        filename = filename + '.npz'
        filename = 'solutions/' + filename

    np.savez(filename,N = N,Nfinal= Nfinal,h = h,ckeep = ckeep,L = L,
    	a = a,b = b,c = c, d = d,lamda = lamda,nu = nu,eta = eta,
    	zeta = zeta, u = ukeep, v = vkeep, t = tkeep)

        
#functions used to compute coefficients before the main loop

def kappa(z):
    return (np.exp(z) - 1)/z

def alpha(z):
    return (-4-z+np.exp(z)*(4-3*z + z**2))/(z**3)

def beta(z):
    return (2 + z + np.exp(z)*(-2 + z))/(z**3)

def gamma(z):
    return (-4 - 3*z - z**2 + np.exp(z)*(4-z))/(z**3)

if __name__ == '__main__':    
    
    a = float(sys.argv[1])
    b = float(sys.argv[2])
    c = float(sys.argv[3])
    d = float(sys.argv[4])
    lamda = float(sys.argv[5])
    nu = float(sys.argv[6])
    eta = float(sys.argv[7])
    zeta = float(sys.argv[8])
    N = int(sys.argv[9])
    h = float(sys.argv[10])
    t_final = float(sys.argv[11])
    number_stored_solutions = int(sys.argv[12])
    L = float(sys.argv[13])
    
    Nfinal = int(t_final/h)
    ckeep = int(Nfinal/number_stored_solutions)
    t_start = time.time()
    #pdb.set_trace()	
    solve_ps(N,Nfinal,h,ckeep,L,a,b,c,d,lamda,nu,eta,zeta)
    t_end = time.time()
    runtime = t_end - t_start
    print 'runtime: %s seconds' % runtime
    
