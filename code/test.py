from __future__ import division
from __future__ import print_function

import os
import glob
import time
import pickle
import random
import argparse
import numpy as np
import torch
from gurobipy import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from EGAT_models import SpGAT

from functools import cmp_to_key


class pair: 
    def __init__(self): 
        self.site = 0
        self.loss = 0

def cmp(a, b):
    if a.loss < b.loss: 
        return -1
    else:
        return 1

def cmp2(a, b):
    if a.loss > b.loss: 
        return -1 
    else:
        return 1 

def Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, now_sol, now_col, constr_flag, lower_bound, upper_bound, value_type):
    '''
    Function Explanation:
    This function solves a problem instance using the SCIP solver based on the provided parameters.

    Parameter Explanation:
    - n: The number of decision variables in the problem instance.
    - m: The number of constraints in the problem instance.
    - k: k[i] indicates the number of decision variables in the i-th constraint.
    - site: site[i][j] indicates which decision variable the j-th decision variable in the i-th constraint is.
    - value: value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    - constraint: constraint[i] represents the right-hand side value of the i-th constraint.
    - constraint_type: constraint_type[i] indicates the type of the i-th constraint, where 1 represents <= and 2 represents >=.
    - coefficient: coefficient[i] indicates the coefficient of the i-th decision variable in the objective function.
    - time_limit: The maximum solving time.
    - obj_type: Specifies whether the problem is a maximization or minimization problem.
    - now_sol: The current solution.
    - now_col: Dimensionality reduction flags for decision variables.
    - constr_flag: Dimensionality reduction flags for constraints.
    - lower_bound: Lower bounds for decision variables.
    - upper_bound: Upper bounds for decision variables.
    - value_type: The type of decision variables (e.g., integer or continuous variables).
    '''
    # Get the start time
    begin_time = time.time()

    # Define the solver model
    model = Model("Gurobi")
    model.feasRelaxS(0,False,False,True)

    # Set up variable mappings
    site_to_new = {}
    new_to_site = {}
    new_num = 0

    # Define new_num decision variables x[]
    x = []
    for i in range(n):
        if(now_col[i] == 1):
            site_to_new[i] = new_num
            new_to_site[new_num] = i
            new_num += 1
            if(value_type[i] == 'B'):
                x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.BINARY))
            elif(value_type[i] == 'C'):
                x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS))
            else:
                x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.INTEGER))

    # Set the objective function and optimization goal (maximize/minimize)
    coeff = 0
    for i in range(n):
        if(now_col[i] == 1):
            coeff += x[site_to_new[i]] * coefficient[i]
        else:
            coeff += now_sol[i] * coefficient[i]
    if(obj_type == 'maximize'):
        model.setObjective(coeff, GRB.MAXIMIZE)
    else:
        model.setObjective(coeff, GRB.MINIMIZE)
    
    # Add m constraints
    for i in range(m):
        if(constr_flag[i] == 0):
            continue
        constr = 0
        flag = 0
        for j in range(k[i]):
            if(now_col[site[i][j]] == 1):
                constr += x[site_to_new[site[i][j]]] * value[i][j]
                flag = 1
            else:
                constr += now_sol[site[i][j]] * value[i][j]

        if(flag == 1):
            if(constraint_type[i] == 1):
                model.addConstr(constr <= constraint[i])
            else:
                model.addConstr(constr >= constraint[i])
        else:
            if(constraint_type[i] == 1):
                if(constr > constraint[i]):
                    print("QwQ")
                    print(constr,  constraint[i])
                    #print(now_col)
            else:
                if(constr < constraint[i]):
                    print("QwQ")
                    print(constr,  constraint[i])
                    #print(now_col)
    
    # Set the maximum solving time
    model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
    
    # Optimize the solution
    model.optimize()
    #print(time.time() - begin_time)
    try:
        new_sol = []
        for i in range(n):
            if(now_col[i] == 0):
                new_sol.append(now_sol[i])
            else:
                if(value_type[i] == 'C'):
                    new_sol.append(x[site_to_new[i]].X)
                else:
                    new_sol.append((int)(x[site_to_new[i]].X))
            
        return new_sol, model.ObjVal
    except:
        return -1, -1

#multilevel_FENNEL
def multilevel_FENNEL(partition_var, partition_num, n, rate, site):
    print(rate, n, int(rate * n))
    parts = [np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)]
    now_num = [0, 0, 0, 0]
    siteA = list(range(0, partition_num))
    random.shuffle(siteA)

    now_color = 0
    for i in range(partition_num):
        now_site = siteA[i]
        now_color = (now_color + 1) % 4
        for var in partition_var[now_site]:
            now_num[now_color] += 1
            parts[now_color][var] = 1
    #print(now_num)
    return(parts)

def cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, solA, blockA, solB, blockB, set_time, constr_flag, lower_bound, upper_bound, value_type):
    crossX = np.zeros(n)
    
    for i in range(n):
        if(blockA[i] == 1):
            crossX[i] = solA[i]
        else:
            crossX[i] = solB[i]
    
    color = np.zeros(n)
    add_num = 0
    for i in range(m):
        if(constr_flag[i] == 0):
            continue
        constr = 0
        flag_A = 0
        flag_B = 0
        number = 0
        for j in range(k[i]):
            if(color[site[i][j]] == 1):
                if(value[i][j] >= 0):
                    flag_A += upper_bound[site[i][j]] * value[i][j]
                    flag_B += lower_bound[site[i][j]] * value[i][j]
                else:
                    flag_B += upper_bound[site[i][j]] * value[i][j]
                    flag_A += lower_bound[site[i][j]] * value[i][j]
            else:
                constr += crossX[site[i][j]] * value[i][j]
            number += 1

        if(constraint_type[i] == 1):
            if(constr + flag_B > constraint[i]):
                #print("1", i, constr, constraint[i])
                for j in range(k[i]):
                    if(color[site[i][j]] == 0):
                        color[site[i][j]] = 1
                        add_num += 1
                        constr -= crossX[site[i][j]] * value[i][j]
                        if(value[i][j] >= 0):
                            flag_A += upper_bound[site[i][j]] * value[i][j]
                            flag_B += lower_bound[site[i][j]] * value[i][j]
                        else:
                            flag_B += upper_bound[site[i][j]] * value[i][j]
                            flag_A += lower_bound[site[i][j]] * value[i][j]
                    if(constr + flag_B <= constraint[i]):
                        break
            
        elif(constraint_type[i] == 2):
            if(constr + flag_A < constraint[i]):
                #print("2", i, constr, constraint[i])
                for j in range(k[i]):
                    if(color[site[i][j]] == 0):
                        color[site[i][j]] = 1
                        add_num += 1
                        constr -= crossX[site[i][j]] * value[i][j]
                        if(value[i][j] >= 0):
                            flag_A += upper_bound[site[i][j]] * value[i][j]
                            flag_B += lower_bound[site[i][j]] * value[i][j]
                        else:
                            flag_B += upper_bound[site[i][j]] * value[i][j]
                            flag_A += lower_bound[site[i][j]] * value[i][j]
                    if(constr + flag_A >= constraint[i]):
                        break
    
    if(add_num / n <= rate):
        newcrossX, newVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, set_time, obj_type, crossX, color, constr_flag, lower_bound, upper_bound, value_type)
        return newcrossX, newVal
    else:
        return -1, -1

# Testing settings
begin_time = time.time()
set_time = 57950
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=32, help='Random seed.')
parser.add_argument('--num', type=int, default=0, help='Test instance num.')
parser.add_argument('--variable_rate', type=float, default=0.3, help='Decision variable dimensionality reduction.')
parser.add_argument('--constraint_rate', type=float, default=0.2, help='Constraint dimensionality reduction.')
parser.add_argument('--model', type=str, default="./86.pkl", help='EGAT Model.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
now_num = args.num
if(os.path.exists('./example-MIP-u/pair' + str(now_num) + '.pickle') == False):
    print("No problem file!")

with open('./example-MIP-u/pair' + str(now_num) + '.pickle', "rb") as f:
    problem = pickle.load(f)

variable_features = problem[0]
constraint_features = problem[1]
edge_indices = problem[2]
edge_feature = problem[3]
#edge, features, labels, idx_train = load_data()

#change
#change
n = len(variable_features)
var_size = len(variable_features[0])
m = len(constraint_features)
con_size = len(constraint_features[0])

edge_num = len(edge_indices[0])

edgeA = []
edgeB = []
edge_features = []
for i in range(edge_num):
    edge_feature[i][0] /= n
for i in range(edge_num):
    edgeA.append([edge_indices[1][i], edge_indices[0][i] + n])
    edgeB.append([edge_indices[0][i] + n, edge_indices[1][i]])
    edge_features.append(edge_feature[i])
edgeA = torch.as_tensor(edgeA)

edgeB = torch.as_tensor(edgeB)


edge_features = torch.as_tensor(edge_features)


for i in range(m):
    for j in range(var_size - con_size):
        constraint_features[i].append(0)
features = variable_features + constraint_features
features = torch.as_tensor(features)


idx_test = torch.tensor(range(n))



if(os.path.exists('./example-MIP-u/data' + str(now_num) + '.pickle') == False):
    print("No problem file!")

with open('./example-MIP-u/data' + str(now_num) + '.pickle', "rb") as f:
    problem = pickle.load(f)

obj_type = problem[0]
n = problem[1]
m = problem[2]
k = problem[3]
site = problem[4]
value = problem[5]
constraint = problem[6]
constraint_type = problem[7]
coefficient = problem[8]
lower_bound = problem[9]
upper_bound = problem[10]
value_type = problem[11]

#print(lower_bound)
#print(upper_bound)
#print(value_type)
##Predict
#FENNEL
partition_num = int(n / 20000)
partition_var = []
for i in range(partition_num):
    partition_var.append([])
vertex_num = n + m
edge_num = 0

edge = []
edge_val = []
for i in range(vertex_num):
    edge.append([])
    edge_val.append([])
for i in range(len(edgeA)):
    edge[edgeA[i][0]].append(edgeA[i][1])
    edge_val[edgeA[i][0]].append(1)
    edge[edgeA[i][1]].append(edgeA[i][0])
    edge_val[edgeA[i][1]].append(1)
    edge_num += 2
       

alpha = (partition_num ** 0.5) * edge_num / (vertex_num ** (2 / 3))
gamma = 1.5
balance = 1.1
#print(alpha)

visit = np.zeros(vertex_num, int)
order = []
for i in range(vertex_num):
    if(visit[i] == 0):
        q = []
        q.append(i)
        visit[i] = 1
        now = 0
        while(now < len(q)):
            order.append(q[now])
            for neighbor in edge[q[now]]:
                if(visit[neighbor] == 0):
                    q.append(neighbor)
                    visit[neighbor] = 1
            now += 1


#print(len(order))
color = np.zeros(vertex_num, int)
for i in range(vertex_num):
    color[i] = -1
cluster_num = np.zeros(partition_num)
score = np.zeros(partition_num, float)
for i in range(vertex_num):
    now_vertex = order[i]
    load_limit = balance * vertex_num / partition_num
    for j in range(len(edge[now_vertex])):
        neighbor = edge[now_vertex][j]
        if(color[neighbor] != -1):
            score[color[neighbor]] += edge_val[now_vertex][j]
    
    now_score = -2e9
    now_site = -1
    for j in range(len(edge[now_vertex])):
        neighbor = edge[now_vertex][j]
        if(color[neighbor] != -1):
            if(score[color[neighbor]] > now_score):
                now_score = score[color[neighbor]]
                now_site = color[neighbor]
    neighbor = random.randint(0, partition_num - 1)
    if(score[neighbor] > now_score):
        now_score = score[neighbor]
        now_site = neighbor
    
    color[now_vertex] = now_site
    score[now_site] += alpha * gamma * (cluster_num[now_site] ** (gamma - 1))
    cluster_num[now_site] += 1
    score[now_site] -= alpha * gamma * (cluster_num[now_site] ** (gamma - 1))
    if(now_vertex < n):
        partition_var[now_site].append(now_vertex - n)

print(color)
print(len(partition_var[0]), len(partition_var[1]), len(partition_var[2]), len(partition_var[3]))
color_site_to_num = []
num_to_color_site = []
color_site_num = []
color_edgeA = []
color_edgeB = []
color_edge_features = []
color_features = []
color_edge_to_num = []
for i in range(partition_num):
    color_site_to_num.append([])
    color_site_num.append(0)
    color_features.append([])
    color_edgeA.append([])
    color_edgeB.append([])
    color_edge_features.append([])
    color_edge_to_num.append([])

for i in range(vertex_num):
    num_to_color_site.append(color_site_num[color[i]])
    color_site_num[color[i]] += 1
    color_site_to_num[color[i]].append(i)
    color_features[color[i]].append(features[i])

edge_num = len(edge_indices[0])
for i in range(edge_num):
    if(color[edge_indices[1][i]] == color[edge_indices[0][i] + n]):
        now_color = color[edge_indices[1][i]]
        color_edgeA[now_color].append([num_to_color_site[edge_indices[1][i]], num_to_color_site[edge_indices[0][i] + n]])
        color_edgeB[now_color].append([num_to_color_site[edge_indices[0][i] + n], num_to_color_site[edge_indices[1][i]]])
        color_edge_features[now_color].append(edge_feature[i])
        color_edge_to_num[now_color].append(i)
print("time:", time.time() - begin_time)

#print(color_edgeA)
#color_edgeA = torch.as_tensor(color_edgeA)
#color_edgeB = torch.as_tensor(color_edgeB)
#color_edge_features = torch.as_tensor(color_edge_features)
#print(color_features)
#color_features = torch.as_tensor(color_features)

path_model = args.model
model = SpGAT(nfeat=features.shape[1],    # Feature dimension
              nhid=64,                    # Feature dimension of each hidden layer
              nclass=2,                   # Number of classes
              dropout=0.5,                # Dropout
              nheads=6,                   # Number of heads
              alpha=0.2)                  # LeakyReLU alpha coefficient
state_dict_load = torch.load(path_model)
#print(state_dict_load)
model.load_state_dict(state_dict_load)
print(model)
model.to(device)


def compute_test(features, edgeA, edgeB, edge_features):
    model.eval()
    output, new_edge_feat = model(features, edgeA, edgeB, edge_features)
    #loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    #acc_test = accuracy(output[idx_test], labels[idx_test])
    #print("Test set results:",
    #      "loss= {:.4f}".format(loss_test.data.item()))
    return(output, new_edge_feat)
predict = []
new_edge_feat = []
for i in range(n + m):
    predict.append([])
for i in range(edge_num):
    new_edge_feat.append(0)
for i in range(partition_num):
    now_predict, now_new_edge_feat = compute_test(torch.tensor([item.cpu().detach().numpy() for item in color_features[i]]).cuda().float().to(device), torch.as_tensor(color_edgeA[i]).to(device), torch.as_tensor(color_edgeB[i]).to(device), torch.as_tensor(color_edge_features[i]).float().to(device))
    for j in range(len(color_site_to_num[i])):
        if(color_site_to_num[i][j] < n):
            predict[color_site_to_num[i][j]] = now_predict[j].cpu().detach().numpy()
    for j in range(len(color_edge_to_num[i])):
        new_edge_feat[color_edge_to_num[i][j]] = now_new_edge_feat[j].cpu().detach().numpy()
#print(predict)       
#print(new_edge_feat)


values = []
for i in range(n):
    values.append(pair())
    values[i].site = i
    values[i].loss = abs(predict[i][0] - 0.5)
    
random.shuffle(values)
values.sort(key = cmp_to_key(cmp))  

constr_dis = []
constr_all = []
constr_flag = []
for i in range(m):
    left = 0
    for j in range(k[i]):
        left += predict[site[i][j]][1] * value[i][j]
    if(constraint_type[i] == 1):
        left = constraint[i] - left
    else:
        left = left - constraint[i]
    constr_dis.append(left)
    constr_all.append(1)
    constr_flag.append(0)
    
L = -10**9
R = 10**9
constr_rate = 0.8
set_constr_rate = args.constraint_rate
num = 0
for i in range(40):
    mid = (L + R) / 2
    now_num = 0
    for j in range(m):
        #print(constr_dis[j],mid)
        if(constr_dis[j] <= mid):
            now_num += 1
    if(now_num / m < constr_rate):
        L = mid
    else:
        R = mid

    #print(now_num / m)

for i in range(m):
    if(constr_dis[i] <= R):
        constr_flag[i] = 1


#print("predict:", predict)
ans_set = []
time_set = []
set_rate = 1.0
rate = args.variable_rate
for turn in range(20):
    obj = (int)(n * (1 - set_rate * rate))

    solution = []
    color = []
    for i in range(n):
        solution.append(0)
        color.append(0)

    numA = 0
    numB = 0
    for i in range(n):
        now_site = values[i].site
        if(i < obj):
            if(predict[now_site][1] < 0.5):
                solution[now_site] = 0
                numA += 1
            else:
                solution[now_site] = 1
                numB += 1
        else:
            color[now_site] = 1
    #print(numA, numB)

    for i in range(m):
        if(constr_flag[i] == 0):
            continue
        constr = 0
        flag_A = 0
        flag_B = 0
        number = 0
        for j in range(k[i]):
            if(color[site[i][j]] == 1):
                if(value[i][j] >= 0):
                    flag_A += upper_bound[site[i][j]] * value[i][j]
                    flag_B += lower_bound[site[i][j]] * value[i][j]
                else:
                    flag_B += upper_bound[site[i][j]] * value[i][j]
                    flag_A += lower_bound[site[i][j]] * value[i][j]
            else:
                constr += solution[site[i][j]] * value[i][j]
            number += 1

        if(constraint_type[i] == 1):
            if(constr + flag_B > constraint[i]):
                #print("1", i, constr, constraint[i])
                for j in range(k[i]):
                    if(color[site[i][j]] == 0):
                        color[site[i][j]] = 1
                        obj -= 1
                        constr -= solution[site[i][j]] * value[i][j]
                        if(value[i][j] >= 0):
                            flag_A += upper_bound[site[i][j]] * value[i][j]
                            flag_B += lower_bound[site[i][j]] * value[i][j]
                        else:
                            flag_B += upper_bound[site[i][j]] * value[i][j]
                            flag_A += lower_bound[site[i][j]] * value[i][j]
                    if(constr + flag_B <= constraint[i]):
                        break
            
        elif(constraint_type[i] == 2):
            if(constr + flag_A < constraint[i]):
                #print("2", i, constr, constraint[i])
                for j in range(k[i]):
                    if(color[site[i][j]] == 0):
                        color[site[i][j]] = 1
                        obj -= 1
                        constr -= solution[site[i][j]] * value[i][j]
                        if(value[i][j] >= 0):
                            flag_A += upper_bound[site[i][j]] * value[i][j]
                            flag_B += lower_bound[site[i][j]] * value[i][j]
                        else:
                            flag_B += upper_bound[site[i][j]] * value[i][j]
                            flag_A += lower_bound[site[i][j]] * value[i][j]
                    if(constr + flag_A >= constraint[i]):
                        break

    print("obj", obj / n)
    if(obj / n + rate >= 1):
        break
    else:
        set_rate -= 0.05
nowX, nowVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, 0.5 * set_time, obj_type, solution, color, constr_flag, lower_bound, upper_bound, value_type)


#LNS
turn_flag = 0
turn = 0
while(time.time() - begin_time < set_time):
    if(constr_rate - 0.01 > set_constr_rate):
        constr_rate -= 0.2
    print("constr_rate: ", constr_rate)
    if(constr_rate < 1):
        constr_dis = []
        constr_flag = []
        for i in range(m):
            left = 0
            for j in range(k[i]):
                left += nowX[site[i][j]] * value[i][j]
            if(constraint_type[i] == 1):
                left = constraint[i] - left
            else:
                left = left - constraint[i]
            constr_dis.append(left)
            constr_flag.append(0)
        L = -10**9
        R = 10**9
        num = 0
        for i in range(30):
            mid = (L + R) / 2
            now_num = 0
            for j in range(m):
                if(constr_dis[j] <= mid):
                    now_num += 1
            if(now_num / m < constr_rate):
                L = mid
            else:
                R = mid

            #print(now_num / m)

        for i in range(m):
            if(constr_dis[i] <= R):
                constr_flag[i] = 1
            
        #Repair
        solution = nowX
        for i in range(n):
            color[i] = 0
        obj = n
        for i in range(m):
            if(constr_flag[i] == 0):
                continue
            constr = 0
            flag = 0
            for j in range(k[i]):
                if(color[site[i][j]] == 1):
                    flag = 1
                else:
                    constr += solution[site[i][j]] * value[i][j]

            if(constraint_type[i] == 1):
                if(constr > constraint[i]):
                    #print("23333")
                    for j in range(k[i]):
                        if(color[site[i][j]] == 0):
                            color[site[i][j]] = 1
                            obj -= 1
            else:
                if(constr + flag < constraint[i]):
                    #print("23333")
                    for j in range(k[i]):
                        if(color[site[i][j]] == 0):
                            color[site[i][j]] = 1
                            obj -= 1
        print("obj", obj / n)
        if(obj / n + rate < 1):
            print("QwQ")
            exit()
        nowX, nowVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, (0.5 * set_time), obj_type, nowX, color, constr_flag, lower_bound, upper_bound, value_type) 
        ans_set.append(nowVal)
        time_set.append(time.time() - begin_time)
    
    
    ##New_FENNEL
    ##Improve
    turnX = []
    turnVal = []
    print("before:", time.time() - begin_time)
    block_list = multilevel_FENNEL(partition_var, partition_num, n, rate, site)
    print("after:", time.time() - begin_time)
    for i in range(4):
        max_time = set_time - (time.time() - begin_time)
        if(max_time <= 0):
            break
        newX, newVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, max_time, obj_type, nowX, block_list[i], constr_flag, lower_bound, upper_bound, value_type)
        if(newVal != -1):
            turnX.append(newX)
            turnVal.append(newVal)
            print("neibor", i, newVal)
    
    #cross
    if(len(turnX) == 4):
        max_time = set_time - (time.time() - begin_time)
        if(max_time <= 0):
            break
        newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[0], block_list[0], turnX[1], block_list[1], max_time, constr_flag, lower_bound, upper_bound, value_type)
        if(newVal != -1):
            turnX.append(newX)
            turnVal.append(newVal)
            print("cross 1", newVal)

        newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[2], block_list[2], turnX[3], block_list[3],  max_time, constr_flag, lower_bound, upper_bound, value_type)
        if(newVal != -1):
            turnX.append(newX)
            turnVal.append(newVal)
            print("cross 2", newVal)
    if(len(turnX) == 6):
        max_time = set_time - (time.time() - begin_time)
        if(max_time <= 0):
            break

        block_list.append(np.zeros(n, int))
        for i in range(n):
            if(block_list[0][i] == 1 or block_list[1][i] == 1):
                block_list[4][i] = 1
        block_list.append(np.zeros(n, int))
        for i in range(n):
            if(block_list[2][i] == 1 or block_list[3][i] == 1):
                block_list[5][i] = 1
        
        newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[4], block_list[4], turnX[5], block_list[5], max_time, constr_flag, lower_bound, upper_bound, value_type)
        if(newVal != -1):
            turnX.append(newX)
            turnVal.append(newVal)
            print("cross 3", newVal)
    
    for i in range(len(turnVal)):
        print(i, ":", turnX[i][:15])
        if(obj_type == 'maximize'):
            if(turnVal[i] > nowVal):
                nowVal = turnVal[i]
                for j in range(n):
                    nowX[j] = turnX[i][j]
        else:
            if(turnVal[i] < nowVal):
                nowVal = turnVal[i]
                for j in range(n):
                    nowX[j] = turnX[i][j]
    
    time_set.append(time.time() - begin_time)
    ans_set.append(nowVal)

#final    
solution = nowX
for i in range(n):
    color[i] = 0
obj = n
for i in range(m):
    constr = 0
    flag = 0
    for j in range(k[i]):
        if(color[site[i][j]] == 1):
            flag = 1
        else:
            constr += solution[site[i][j]] * value[i][j]

    if(constraint_type[i] == 1):
        if(constr > constraint[i]):
            #print("23333")
            for j in range(k[i]):
                if(color[site[i][j]] == 0):
                    color[site[i][j]] = 1
                    obj -= 1
                    constr -= solution[site[i][j]] * value[i][j]
    else:
        if(constr + flag < constraint[i]):
            #print("23333")
            for j in range(k[i]):
                if(color[site[i][j]] == 0):
                    color[site[i][j]] = 1
                    obj -= 1
print("obj", obj / n)
if(obj / n + rate < 1):
    print("QwQ")
    exit()
nowX, nowVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, (0.5 * set_time), obj_type, solution, color, constr_all, lower_bound, upper_bound, value_type) 
ans_set.append(nowVal)
time_set.append(time.time() - begin_time)

print(ans_set)
print(time_set)




