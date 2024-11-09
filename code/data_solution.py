from gurobipy import *
import numpy as np
import argparse
import pickle
import random
import time
import os


def Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, lower_bound, upper_bound, value_type):
    '''
    Function description:
    Solves a problem instance using the Gurobi solver based on the provided inputs.

    Parameter descriptions:
    - n: Number of decision variables in the problem instance.
    - m: Number of constraints in the problem instance.
    - k: k[i] indicates the number of decision variables involved in the ith constraint.
    - site: site[i][j] indicates which decision variable is involved in the jth position of the ith constraint.
    - value: value[i][j] indicates the coefficient of the jth decision variable in the ith constraint.
    - constraint: constraint[i] indicates the right-hand side value of the ith constraint.
    - constraint_type: constraint_type[i] indicates the type of the ith constraint, where 1 represents <= and 2 represents >=.
    - coefficient: coefficient[i] represents the coefficient of the ith decision variable in the objective function.
    - time_limit: Maximum time allowed for solving.
    - obj_type: Indicates whether the problem is a maximization or minimization problem.
    - lower_bound: lower_bound[i] represents the lower bound of the range for the ith decision variable.
    - upper_bound: upper_bound[i] represents the upper bound of the range for the ith decision variable.
    - value_type: value_type[i] represents the type of the ith decision variable, 'B' indicates a binary variable, 'I' indicates an integer variable, 'C' indicates a continuous variable.
    '''

    # Get the start time
    begin_time = time.time()
    # Define the optimization model
    model = Model("Gurobi")
    # Define n decision variables x[]
    x = []
    for i in range(n):
        if(value_type[i] == 'B'):
            x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.BINARY))
        elif(value_type[i] == 'C'):
            x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS))
        else:
            x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.INTEGER))
    # Set the objective function and optimization goal (maximize/minimize)
    coeff = 0
    for i in range(n):
        coeff += x[i] * coefficient[i]
    if(obj_type == 'maximize'):
        model.setObjective(coeff, GRB.MAXIMIZE)
    else:
        model.setObjective(coeff, GRB.MINIMIZE)
    # Add m constraints
    for i in range(m):
        constr = 0
        for j in range(k[i]):
            #print(i, j, k[i])
            constr += x[site[i][j]] * value[i][j]
        if(constraint_type[i] == 1):
            model.addConstr(constr <= constraint[i])
        elif(constraint_type[i] == 2):
            model.addConstr(constr >= constraint[i])
        else:
            model.addConstr(constr == constraint[i])
    # Set the maximum solving time
    model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
    # Optimize the solution
    model.optimize()
    ans = []
    for i in range(n):
        if(value_type[i] == 'C'):
            ans.append(x[i].X)
        else:
            ans.append(int(x[i].X))
    return ans



def optimize(
    time: int,
    number: int,
):
    '''
    Function description:
    Designs and calls a specified algorithm and solver to optimize the problem stored in data.pickle in the same directory, based on the provided parameters.

    Parameter descriptions:
    - time: Time allotted per instance for Gurobi to find reference solutions.
    - number: Integer type, indicates the number of instances to be generated.
    '''

    for num in range(number):
        # Check if data.pickle exists; if it doesn't, read it
        if not os.path.exists('./example/data' + str(num) + '.pickle'):
            print("No input file!")
            return 
        with open('./example/data' + str(num) + '.pickle', "rb") as f:
            data = pickle.load(f)
    
        # n represents the number of decision variables
        # m represents the number of constraints
        # k[i] represents the number of decision variables in the ith constraint
        # site[i][j] represents the decision variable in the jth position of the ith constraint
        # value[i][j] represents the coefficient of the jth decision variable in the ith constraint
        # constraint[i] represents the right-hand side value of the ith constraint
        # constraint_type[i] represents the type of the ith constraint, where 1 is <=, 2 is >=
        # coefficient[i] represents the coefficient of the ith decision variable in the objective function
        # lower_bound[i] represents the lower bound of the range for the ith decision variable.
        # upper_bound[i] represents the upper bound of the range for the ith decision variable.
        # value_type[i] represents the type of the ith decision variable, 'B' for binary variable, 'I' for integer variable, 'C' for continuous variable.
        n = data[1]
        m = data[2]
        k = data[3]
        site = data[4]
        value = data[5]
        constraint = data[6]
        constraint_type = data[7]
        coefficient = data[8]
        lower_bound = data[9]
        upper_bound = data[10]
        value_type = data[11]
        obj_type = data[0]
        
        optimal_solution = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time, obj_type, lower_bound, upper_bound, value_type)


        # Bipartite graph encoding
        variable_features = []
        constraint_features = []
        edge_indices = [[], []] 
        edge_features = []

        #print(value_type)
        for i in range(n):
            now_variable_features = []
            now_variable_features.append(coefficient[i])
            now_variable_features.append(0)
            now_variable_features.append(1)
            if(value_type[i] == 'C'):
                now_variable_features.append(0)
            else:
                now_variable_features.append(1)
            now_variable_features.append(random.random())
            variable_features.append(now_variable_features)
        
        for i in range(m):
            now_constraint_features = []
            now_constraint_features.append(constraint[i])
            if(constraint_type[i] == 1):
                now_constraint_features.append(1)
                now_constraint_features.append(0)
                now_constraint_features.append(0)
            if(constraint_type[i] == 2):
                now_constraint_features.append(0)
                now_constraint_features.append(1)
                now_constraint_features.append(0)
            if(constraint_type[i] == 3):
                now_constraint_features.append(0)
                now_constraint_features.append(0)
                now_constraint_features.append(1)
            now_constraint_features.append(random.random())
            constraint_features.append(now_constraint_features)
        
        for i in range(m):
            for j in range(k[i]):
                edge_indices[0].append(i)
                edge_indices[1].append(site[i][j])
                edge_features.append([value[i][j]])

        with open('./example/pair' + str(num) + '.pickle', 'wb') as f:
                pickle.dump([variable_features, constraint_features, edge_indices, edge_features, optimal_solution], f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type = int, default = 10, help = 'Running wall-clock time.')
    parser.add_argument("--number", type = int, default = 10, help = 'The number of instances.')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    #print(vars(args))
    optimize(**vars(args))