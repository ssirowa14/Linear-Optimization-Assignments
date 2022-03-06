#ROll numbers of team members:
#1) ES16BTECH11009 - Devansh Agarwal
#2) CS16BTECH11030 - Saahil Sirowa
#3) EE16BTECH11042 - Anand N Warrier

import numpy as np
import sympy as sp
import sys

def getInput():
	m = int(input("Enter m which is the number of rows ")) 
	n = int(input("Enter n which is the number of columns ")) 
	A = []
	print("Enter Matrix A\n", "Enter one row at a time\n for example- 1 2 3 <enter> 4 5 6 <enter>\n","for matrix:\n","1 2 3\n","4 5 6")
	for i in range(m):
		A.append(list(map(float, input().split())))
	print("Enter vector B\n", "example:- 1 2 3<enter>\n", "for vector: 1 2 3")
	B = list(map(float, input().split()))
	print("Enter vector C\n", "example:- 1 2 3<enter>\n", "for vector: 1 2 3")
	C = list(map(float, input().split()))
	return np.array(A), np.array(B), np.array(C)

def simplex(A,b,c,initial_point,print_details_flag=False):
    # Objective: maximize c*x s.t. Ax<=b
    m=len(A)    # Number of rows of A
    n=len(A[0]) # Number of columns of A
    
    # adding the xi>=0 constraints
    b=np.concatenate((b,np.zeros(n)))
    A=np.concatenate((A,-1*np.identity(n)),axis=0)
    x=initial_point # we start from the given initital point

    # For finding the equations that become tight at x, i.e., when A[i]*x=b[i]
    tight_eqns_index_list=[i for i in range(len(A)) if abs(np.dot(A[i],x) - b[i])< 0.00000001]
    untight_eqns_index_list=[i for i in range(len(A)) if abs(np.dot(A[i],x) - b[i])> 0.00000001]

    A_tight=A[tight_eqns_index_list] # matrix of tight equations
    b_tight=b[tight_eqns_index_list] # and corresponding b[i]'s

    # Finding the indices of the linearly independent rows of the A_tight
    coeffs,indices=sp.Matrix(A_tight).T.rref()
    indices=sorted(indices)
    complement_indices=[i for i in range(len(A_tight)) if i not in indices]

    # A_dash is the matrix of n linearly independent rows of A that are tight at x
    # A_dash_dash consists of all other rows
    A_dash=A_tight[indices]
    A_dash_dash=A_tight[complement_indices]
    A_dash_dash=np.concatenate((A_dash_dash,A[untight_eqns_index_list]))

    # b_dash corresponds to the b[i]'s of the constraints represented by the rows of A_dash
    # b_dash_dash corresponds to the b[i]'s of the constraints represented by the rows of A_dash_dash
    b_dash=b_tight[indices]
    b_dash_dash=b_tight[complement_indices]
    b_dash_dash=np.concatenate((b_dash_dash,b[untight_eqns_index_list]))

    iter=0
    while True:
        if iter==10000:
            print("\nBREAKING AT THE 10000TH ITERATION TO AVOID INFINITE LOOPING!...")
            raise
        if print_details_flag:
            print("\nStart of iteration ",iter,", Optimal value:",np.dot(c,x))

        try:
            z=-1*np.linalg.inv(A_dash)  # columns of z give the direction vectors of each neighbour
        except:
            print("\nUnbounded problem!")
            exit()
        z_T=z.T # z_T is just the z Transpose, created to iterate over columns
        k=0
        while k<len(z_T): 
            Bk=z_T[k] # Bk is the kth column of z or the kth row of z_T
            c_dot_Bk=np.dot(c,Bk)
            if c_dot_Bk >0: # choose the direction(given by Bk) for which this dot product is >0
                break
            k+=1
        if k==len(z_T): # No such Bk was found => reached optimum
            if print_details_flag:
                print("Reached optimum!")
            break
        Bk=z_T[k] # the direction of increase
        """
        We loop to find the minimum of (b[s]-A[s]*x)/A[s]*Bk,
        where s ranges over all rows of A that belong to A_dash_dash
        """
        s=0
        t=sys.float_info.max 
        all_denom_critical=True # for checking whether np.dot(A_dash_dash[i],Bk) is <=0 for all i
        for i in range(len(A_dash_dash)):
            denom=np.dot(A_dash_dash[i],Bk)
            if denom>0: # ignore otherwise
                all_denom_critical=False
                temp=(b_dash_dash[i]-np.dot(A_dash_dash[i],x))/denom
                if temp<t : 
                    t=temp 
                    s=i
        if all_denom_critical or t>=10e6 : # when all np.dot(A_dash_dash[i],Bk)<=0 or t explodes
            print("\nUnbounded problem!")
            exit()
        x=x+t*Bk # Our new point

        # backing up the kth row of A_dash and the sth row A_dash_dash before removing them
        row_k=A_dash[k] 
        row_s=A_dash_dash[s]

        A_dash=A_dash[[j for j in range(len(A_dash)) if j!=k]] # removing the kth row of A_dash
        A_dash=np.concatenate((A_dash,np.array([row_s]))) # putting the sth row of A_dash_dash into A_dash

        A_dash_dash=A_dash_dash[[j for j in range(len(A_dash_dash)) if j!=s]] # removing the sth row of A_dash_dash
        A_dash_dash=np.concatenate((A_dash_dash,np.array([row_k]))) # putting the kth row of A_dash into A_dash_dash

        # Doing the same thing with the b vector
        # backing up the kth element of b_dash and the sth element of b_dash_dash
        b_dash_k=b_dash[k] 
        b_dash_dash_s=b_dash_dash[s]

        b_dash=b_dash[[i for i in range(len(b_dash)) if i!=k]] # removing the kth element of b_dash
        b_dash=np.concatenate((b_dash,np.array([b_dash_dash_s]))) # putting the sth element of b_dash_dash into b_dash 

        b_dash_dash=b_dash_dash[[i for i in range(len(b_dash_dash)) if i!=s]] # removing the sth element of b_dash_dash
        b_dash_dash=np.concatenate((b_dash_dash,np.array([b_dash_k]))) # putting the kth element of b_dash into b_dash_dash
        iter+=1
    if print_details_flag == True:  
        return x, A_dash, b_dash, A, b
    else:
        return x

def main():
    A,b,c=getInput()
    bOriginal = np.array(b)
    AOriginal = np.array(A)
    cOriginal = np.array(c)
    while 1: #loop till degeneracy is removed
        try:
            A = np.array(AOriginal)
            b = np.array(bOriginal)
            c = np.array(cOriginal)
            for i in range(len(b)):             #add noise to remove degeneracy
                epsilon = np.random.uniform(low=1e-6, high=1e-5)
                b[i] = b[i] + epsilon

            m=len(A)
            n=len(A[0])
            #checking for negative entries in b
            min_b=sys.float_info.max
            for i in range(m):
                if b[i]<0 and b[i]<min_b:
                    min_b=b[i]

            initial_point=np.zeros(n) # Initial point is origin unless there is a negative entry in b
            if min_b<0: # if there is a negative entry in b
                print("\nNegative entry in b detected. Phase 1 starts...")
                # the suffix "_hat" of variables indicates that they are used for phase 1
                # the problem is modified by adding another variable z to system 
                A_hat=np.vstack((A,np.zeros(n))) 
                ones_arr=np.ones((m+1,1))
                ones_arr[m]=-1
                A_hat=np.append(A_hat,-1*ones_arr,axis=1)
                c_hat=np.concatenate((np.zeros(n),[-1]))
                b_hat=np.concatenate((b,[-min_b]))
                initial_point_hat=np.concatenate((np.zeros(n),[-min_b]))
                x_hat=simplex(A_hat,b_hat,c_hat,initial_point_hat)
                z=np.dot(c_hat,x_hat)
                if z<0:
                    print("Problem is Infeasible!")
                    exit()
                else:
                    initial_point=x_hat[:n] 
                print("Feasible point found:",initial_point)
                print("\nPhase 2 begins...")
            x, A_dash, b_dash, A, b=simplex(A,b,c,initial_point,True)       #x is the solution is noise added b
            try:
                b_final = []
                for i in range(len(A_dash)):                                #finding original tight constraints to the solution
                    for j in range(len(A)):
                        if np.array_equal(A_dash[i,:],A[j,:]) == True and b_dash[i] == b[j]:
                            if j < len(bOriginal):
                                b_final.append(bOriginal[j])
                            else:
                                b_final.append(0)
                            break
                b_final = np.array(b_final)             #rhs
                solve_point = np.linalg.solve(A_dash,b_final)   #find intersection of constraints to find point corresponding to original problem
            except Exception as e:
                solve_point = x
            print("Optimum value:",np.round(np.dot(c,solve_point), 4),"\nSolution vector:",np.round(x, 4))
            break
        except Exception as e:
            continue
if __name__=="__main__":
    main()
