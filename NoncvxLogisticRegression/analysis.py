########################################################################################################################
####---------------------------------------------------Analysis-----------------------------------------------------####
########################################################################################################################

## Used to calculate different types of errors for all algorithms

import numpy as np
from numpy import linalg as LA

class error:
    def __init__(self, problem, model_optimal, cost_optimal):
        self.pr = problem                                       ## problem class
        self.N = self.pr.N                                      ## total number of data samples
        self.X = self.pr.X_train                                ## feature vectors
        self.Y = self.pr.Y_train                                ## label vector             
        self.theta_opt = model_optimal
        self.F_opt = cost_optimal
        self.n = problem.n
        
    def path_cls_error(self, iterates):
        iterates = np.array( iterates )          
        Y_predict = np.matmul( self.X, iterates.T )
        error_matrix = np.multiply( Y_predict, self.Y[:,np.newaxis] ) < 0
        return np.sum( error_matrix, axis = 0 ) / self.N

    def point_cls_error(self, theta):
        Y_predict = np.matmul( self.X, theta )
        error = Y_predict * self.Y < 0 
        return sum(error)/self.N

    def theta_gap_path(self, thetas):
        path = []
        for theta in thetas:
            path.append(LA.norm(theta - self.theta_opt, 'fro')**2 / self.n)
        return path

    def ctheta_gap_path(self, thetas):
        path = []
        for theta in thetas:
            path.append(LA.norm(theta - self.theta_opt, 2)**2)
        return path

    def cost_gap_point(self, theta):
        return self.pr.F_val(theta) - self.F_opt
    
    def cost_gap_path(self, iterates):
        K = len(iterates)
        result = [ ]
        for k in range(K):
            result.append( error.cost_gap_point(self,iterates[k]) )
        return result

    def ngrad_point(self, theta):
        return LA.norm(self.pr.grad(theta))**2

    def ngrad_path(self, iterates):
        K = len(iterates)
        result = []
        for k in range(K):
            result.append(self.ngrad_point(iterates[k]))
        return result

    def ngrad_point_cons(self, Theta):
        grad = self.pr.networkgrad(Theta)
        return LA.norm(np.mean(grad, 0))**2

    def ngrad_path_cons(self, iterates):
        K = len(iterates)
        result = []
        for k in range(K):
            result.append(self.ngrad_point_cons(iterates[k]))
        return result

    def ngrad_point_both(self, Theta):
        grad = self.pr.networkgrad(Theta)
        grad_mean = self.pr.grad(np.sum(Theta, axis=0) / self.pr.n)
        return LA.norm(np.mean(grad, 0))**2 / 2 + LA.norm(grad_mean)**2 / 2

    def ngrad_path_both(self, iterates):
        K = len(iterates)
        result = []
        for k in range(K):
            result.append(self.ngrad_point_both(iterates[k]))
        return result







