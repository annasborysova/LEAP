# -*- coding: utf-8 -*-
"""
Created on Sun Sep 03 07:36:56 2017

@author: Anna
"""
import glob, os


def get_average(param, clf):
    avg = 0    
    print("average for {}, param {}".format(clf, param))
    file_num = 0
    for logfile in glob.glob(os.path.join(".", "logs", "*.txt")):
        include = False
        for i, line in enumerate(open(logfile)):
            if "chosen features" in line and len(line) > 24 and clf in line and param in line:
                include = True
            if "CLASSIFIER: " + clf in line and include:
                avg += float(line.split()[-1])
                continue
        if include:
            file_num += 1
                
    print(file_num)
    print avg / file_num


for logfile in glob.glob(os.path.join(".", "logs", "*.txt")):
    for i, line in enumerate(open(logfile)):
        if "chosen features" in line and len(line) > 24 and "MLP" in line:
            print(line)

get_average('tanh', 'MLP')   
get_average('logistic', 'MLP')
get_average('relu', 'MLP')
get_average('identity', 'MLP')

#get_average('constant', 'MLP')   
#get_average('invscaling', 'MLP')
#get_average('adaptive', 'MLP')




#
#
#
#ball_av = 0
#print("adam")   
#for logfile in ball_trees:
#    for i, line in enumerate(open(logfile)):
#        if "CLASSIFIER: MLP" in line:
#            ball_av += float(line.split()[-1])         
#
#print(len(ball_trees))
#ball_av = ball_av / len(ball_trees)
#print ball_av


#algorithm ball_tree: 0.365989010989 average
# algorithm auto: 0.389749492642 average
# algorithms inconclusive

#rbf
#22
#0.515135164353
#Poly
#5
#0.510113297555
#Linear
#3
#0.46282051282
# choose rbf

#ovo
#23
#0.520079806292
#ovr
#7
#0.472880871043
# choose ovo

#sgd
#8
#0.487019230769
#adam
#6
#0.475641025641
#inconclusive

#average for MLP, param constant
#10
#0.500952977533
#average for MLP, param invscaling
#8
#0.497424608317
#average for MLP, param adaptive
#8
#0.474038461538
#inconclusive

#average for MLP, param tanh
#9
#0.480189395427
#average for MLP, param logistic
#7
#0.554108649222
#average for MLP, param relu
#5
#0.466923076923
#average for MLP, param identity
#5
#0.449230769231
#suspect

# the only kNN uniform weights which was trainined on more than one participant gave terrible performance .\logs\experiment_1503775988.14.txt
