"""
This python file contains the necessarry functions and varriables to solve the problem
For I like to separate functions, I decided to create this file next to my notebook
The other reasen was, that I use VSC with some quite handy extensions and most of them work only in python files and do not in the notebooks
"""

# Importing the necessarry libraries
import numpy as np
import matplotlib.pyplot as plt

########################################################

# Creating the varriables:
h = .1
n_1 = 1000
n_2 = 130
t0 = 0
t_max = 100
######################################################


# Creating the functions to solve the problem
def f(n,a,b,c,d):
    # In this function I just return both population change in one numpy array
    return np.array([a*n[0]-b*n[1] *n[0], c*n[0] *n[1]-d*n[1]])

"""
Although scipy.integrate implemented the runge-kutta, I felt it more convinient to write my own
Which is specific for the task
"""
def rk4(x,*args):
    # Runge kutta solver I guess not need any explanation
    k1 = f(x,*args)
    k2 = f(x+0.5*h*k1,*args)
    k3 = f(x+0.5*h*k2,*args)
    k4 = f(x+k3*h,*args)

    return x + h/6 * (k1 + 2* k2 + 2*k3+k4)

def calculate(*args):
    # This is where I calculate populations and the time

    t_list = [t0]
    n_list = [np.array([n_1,n_2])]
    t = t0
    for _ in range( int(t_max/h) ):
        n_list.append(rk4(n_list[-1],*args))
        t +=1*h
        t_list.append(t)
    return np.array(n_list), t_list

def lv(a,b,c,d):
    """
    This function is just creates some plots
    First I calculate the result of the Lotka-Voltera equations
    Then plot the results
    """
    n_list,t_list = calculate(a,b,c,d)

    plt.figure(figsize=[16,8])
    plt.subplot(121)
    plt.plot(n_list[:,0], n_list[:,1],label=r"$\alpha$="+str(a)+"\n"+r"$\beta$="+str(b)+"\n"+r"$\gamma$="+str(c)+"\n"+r"$\delta$="+str(d),color="tab:green")
    plt.legend(fontsize=16)
    plt.grid()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params("both",length=12,width=2)
    plt.tick_params("both",which="minor",length=8,width=2)
    plt.xlabel("Number of preys[1]",size=16)
    plt.ylabel("Number of predators[1]",size=16)

    plt.subplot(122)
    plt.plot(t_list,n_list[:,0], label="prey:\n"+ r"$\alpha$="+str(a)+"\n$" + r"\beta$="+str(b),color="tab:green")
    plt.plot(t_list,n_list[:,1], label="predator:\n"+r"$\gamma$="+str(c)+"\n$"+r"\delta$="+str(d),color="tab:orange")
    plt.legend(fontsize=16)
    plt.grid()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("time[1]",size=16)
    plt.ylabel("Number of species[1]",size=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params("both",length=12,width=2)
    plt.tick_params("both",which="minor",length=8,width=2)

    plt.tight_layout()
    plt.savefig("pics/2.png")

def f_2(n,r1,r2,a,b,k1,k2):
    """
    This function returns a numpy array
    x is the change of the first specie
    y is the change of the second specie
    """
    x = r1*n[0]*(1- (n[0]+a*n[1])/k1)
    y = r2*n[1]*(1- (n[1]+b*n[0])/k2)
    return np.array([x,y])

def rk4_2(x,*args):
    """
    This is my modified Runge-Kutta specificly for the competitive Lotka-Volterra
    """
    k1 = f_2(x,*args)
    k2 = f_2(x+0.5*h*k1,*args)
    k3 = f_2(x+0.5*h*k2,*args)
    k4 = f_2(x+k3*h,*args)

    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)


def calculate_2(*args):
    t_list = [0]
    n_list = [np.array([4,2])]
    t = t0
    for _ in range(int(t_max/h)):
        n_list.append(rk4_2(n_list[-1], *args))
        t += 1
        t_list.append(t)
    return np.array(n_list), t_list

def compete(r1,r2,a,b,k1,k2,picID):
    n_list, t_list = calculate_2(r1,r2,a,b,k1,k2)

    plt.figure(figsize=[16,8])

    plt.plot(t_list,n_list[:,0],label="First specie($r_1,a,k_1$): ("+str(r1)+","+str(a)+","+str(k1)+")",color="tab:green",lw = 4)
    plt.plot(t_list,n_list[:,1],label="Second specie($r_2,b,k_2$): ("+str(r2)+","+str(b)+","+str(k2)+")",color="tab:orange",lw = 4)
    plt.xlabel("Time [1]",size=20)
    plt.ylabel("Population [1]",size=20)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params("both",length=12,width=2)
    plt.tick_params("both",which="minor",length=8,width=2)
    plt.grid()

    plt.savefig("pics/"+str(picID)+".png")

def real_life_plot(df):
    plt.figure(figsize=[16,8])

    plt.plot(df['Year'],df[df.columns[1]],Label = df.columns[1], color = "green")
    plt.plot(df['Year'],df[df.columns[2]],Label = df.columns[2],color = "red")

    plt.title('Population changes on real species',fontsize = 22)
    plt.xlabel("Year", fontsize = 22)
    plt.ylabel("Population", fontsize = 22)

    plt.legend(fontsize=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.savefig('pics/ref1')