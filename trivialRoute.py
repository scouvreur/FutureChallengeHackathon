# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# import h5py

# citydata = pd.read_csv('CityData.csv')

# train = pd.read_csv('ForecastDataforTraining_201712.csv') #4GB

# train

# test = pd.read_csv('ForecastDataforTesting_201712.csv') #4GB

# labeldata = pd.read_csv('In_situMeasurementforTraining_201712.csv')

# citydata

balloon_routes = {}

balloon_routes = {}
# b1d1 = {}
# key = 0
# b1d1[key] = 0
# for x in range(142,83,-1):
#     b1d1[key]=(x,328)
#     key += 1
# for y in range(328,202,-1):
#     b1d1[key]=(84,y)
#     key += 1
# balloon_routes[(1,6)] = b1d1
# balloon_routes[(1,7)] = b1d1
# balloon_routes[(1,8)] = b1d1
# balloon_routes[(1,9)] = b1d1
# balloon_routes[(1,10)] = b1d1

# balloon_routes[(1,10)]

#
b1d1 = {}
key = 0
b1d1[key] = 0
originx = 142
originy = 328
destx=84
desty=203

if destx>originx :
    xstp = 1
    destx +=1
else :
    xstp = -1
    destx -=1

if desty>originy :
    ystp = 1
    desty += 1
else :
    ystp = -1
    desty -= 1

for x in range(originx,destx,xstp):
    b1d1[key]=(x,originy)
    key += 1
for y in range(originy+ystp,desty,ystp):
    b1d1[key]=(destx-xstp,y)
    key += 1

balloon_routes[(1,6)] = b1d1
balloon_routes[(1,7)] = b1d1
balloon_routes[(1,8)] = b1d1
balloon_routes[(1,9)] = b1d1
balloon_routes[(1,10)] = b1d1

balloon_routes[(1,6)]
#

b2d1 = {}
key = 0
b2d1[key] = 0
originx = 142
originy = 328
destx=199
desty=371

if destx>originx :
    xstp = 1
    destx +=1
else :
    xstp = -1

if desty>originy :
    ystp = 1
    desty += 1
else :
    ystp = -1
    
for x in range(originx,destx,xstp):
    b2d1[key]=(x,originy)
    key += 1
for y in range(originy+1,desty,ystp):
    b2d1[key]=(destx-1,y)
    key += 1

b2d1
balloon_routes[(2,6)] = b2d1
balloon_routes[(2,7)] = b2d1
balloon_routes[(2,8)] = b2d1
balloon_routes[(2,9)] = b2d1
balloon_routes[(2,10)] = b2d1

balloon_routes[(2,10)]

b3d1 = {}
key = 0
b3d1[key] = 0
originx = 142
originy = 328
destx=140
desty=234

if destx>originx :
    xstp = 1
    destx +=1
else :
    xstp = -1
    destx -=1

if desty>originy :
    ystp = 1
    desty += 1
else :
    ystp = -1
    desty -= 1

for x in range(originx,destx,xstp):
    b3d1[key]=(x,originy)
    key += 1
for y in range(originy+ystp,desty,ystp):
    b3d1[key]=(destx-ystp,y)
    key += 1

balloon_routes[(3,6)] = b3d1
balloon_routes[(3,7)] = b3d1
balloon_routes[(3,8)] = b3d1
balloon_routes[(3,9)] = b3d1
balloon_routes[(3,10)] = b3d1

balloon_routes[(3,10)]

b4d1 = {}
key = 0
b4d1[key] = 0
originx = 142
originy = 328
destx=236
desty=241

if destx>originx :
    xstp = 1
    destx +=1
else :
    xstp = -1
    destx -=1

if desty>originy :
    ystp = 1
    desty += 1
else :
    ystp = -1
    desty -= 1

for x in range(originx,destx,xstp):
    b4d1[key]=(x,originy)
    key += 1
for y in range(originy+ystp,desty,ystp):
    b4d1[key]=(destx-xstp,y)
    key += 1

balloon_routes[(4,6)] = b4d1
balloon_routes[(4,7)] = b4d1
balloon_routes[(4,8)] = b4d1
balloon_routes[(4,9)] = b4d1
balloon_routes[(4,10)] = b4d1

balloon_routes[(4,10)]

b5d1 = {}
key = 0
b5d1[key] = 0
originx = 142
originy = 328
destx=315
desty=281

if destx>originx :
    xstp = 1
    destx +=1
else :
    xstp = -1
    destx -=1

if desty>originy :
    ystp = 1
    desty += 1
else :
    ystp = -1
    desty -= 1

for x in range(originx,destx,xstp):
    b5d1[key]=(x,originy)
    key += 1
for y in range(originy+ystp,desty,ystp):
    b5d1[key]=(destx-xstp,y)
    key += 1

balloon_routes[(5,6)] = b5d1
balloon_routes[(5,7)] = b5d1
balloon_routes[(5,8)] = b5d1
balloon_routes[(5,9)] = b5d1
balloon_routes[(5,10)] = b5d1

balloon_routes[(5,10)]

b6d1 = {}
key = 0
b6d1[key] = 0
originx = 142
originy = 328
destx=358
desty=207

if destx>originx :
    xstp = 1
    destx +=1
else :
    xstp = -1
    destx -=1

if desty>originy :
    ystp = 1
    desty += 1
else :
    ystp = -1
    desty -= 1

for x in range(originx,destx,xstp):
    b6d1[key]=(x,originy)
    key += 1
for y in range(originy+ystp,desty,ystp):
    b6d1[key]=(destx-xstp,y)
    key += 1

balloon_routes[(6,6)] = b6d1
balloon_routes[(6,7)] = b6d1
balloon_routes[(6,8)] = b6d1
balloon_routes[(6,9)] = b6d1
balloon_routes[(6,10)] = b6d1

balloon_routes[(6,10)]

b7d1 = {}
key = 0
b7d1[key] = 0
originx = 142
originy = 328
destx=363
desty=237

if destx>originx :
    xstp = 1
    destx +=1
else :
    xstp = -1
    destx -=1

if desty>originy :
    ystp = 1
    desty += 1
else :
    ystp = -1
    desty -= 1

for x in range(originx,destx,xstp):
    b7d1[key]=(x,originy)
    key += 1
for y in range(originy+ystp,desty,ystp):
    b7d1[key]=(destx-xstp,y)
    key += 1

balloon_routes[(7,6)] = b7d1
balloon_routes[(7,7)] = b7d1
balloon_routes[(7,8)] = b7d1
balloon_routes[(7,9)] = b7d1
balloon_routes[(7,10)] = b7d1

balloon_routes[(7,10)]

b8d1 = {}
key = 0
b8d1[key] = 0
originx = 142
originy = 328
destx=423
desty=266

if destx>originx :
    xstp = 1
    destx +=1
else :
    xstp = -1
    destx -=1

if desty>originy :
    ystp = 1
    desty += 1
else :
    ystp = -1
    desty -= 1

for x in range(originx,destx,xstp):
    b8d1[key]=(x,originy)
    key += 1
for y in range(originy+ystp,desty,ystp):
    b8d1[key]=(destx-xstp,y)
    key += 1

balloon_routes[(8,6)] = b8d1
balloon_routes[(8,7)] = b8d1
balloon_routes[(8,8)] = b8d1
balloon_routes[(8,9)] = b8d1
balloon_routes[(8,10)] = b8d1

balloon_routes[(8,10)]

b9d1 = {}
key = 0
b9d1[key] = 0
originx = 142
originy = 328
destx=125
desty=375

if destx>originx :
    xstp = 1
    destx +=1
else :
    xstp = -1
    destx -=1

if desty>originy :
    ystp = 1
    desty += 1
else :
    ystp = -1
    desty -= 1

for x in range(originx,destx,xstp):
    b9d1[key]=(x,originy)
    key += 1
for y in range(originy+ystp,desty,ystp):
    b9d1[key]=(destx-xstp,y)
    key += 1

balloon_routes[(9,6)] = b9d1
balloon_routes[(9,7)] = b9d1
balloon_routes[(9,8)] = b9d1
balloon_routes[(9,9)] = b9d1
balloon_routes[(9,10)] = b9d1

balloon_routes[(9,10)]

b10d1 = {}
key = 0
b10d1[key] = 0
originx = 142
originy = 328
destx=189
desty=274

if destx>originx :
    xstp = 1
    destx +=1
else :
    xstp = -1
    destx -=1

if desty>originy :
    ystp = 1
    desty += 1
else :
    ystp = -1
    desty -= 1

for x in range(originx,destx,xstp):
    b10d1[key]=(x,originy)
    key += 1
for y in range(originy+ystp,desty,ystp):
    b10d1[key]=(destx-xstp,y)
    key += 1

balloon_routes[(10,6)] = b10d1
balloon_routes[(10,7)] = b10d1
balloon_routes[(10,8)] = b10d1
balloon_routes[(10,9)] = b10d1
balloon_routes[(10,10)] = b10d1

balloon_routes[(10,10)]

for balloon in range(1,11):
    for day in range(6,11):
        key = 0
        for h in range (3,22):
            if (h!=21):
                for m in range(0,60,2):
                    if key >= len(balloon_routes[(balloon,day)]) : continue 
                    print (balloon,day,'{:02d}:{:02d}'.format(h,m),balloon_routes[(balloon,day)][key][0],balloon_routes[(balloon,day)][key][1],sep=',')
                    key += 1
            else:
                if key >= len(balloon_routes[(balloon,day)]) : continue
                print(balloon,day,h,"21:00",balloon_routes[(balloon,day)][key][0],balloon_routes[(balloon,day)][key][1],sep=',')
