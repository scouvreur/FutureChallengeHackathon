import datetime
import h5py

def manhatten_distance(a,b):
    #a,b should be a tuple with (x,y) as coordinate
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def move(pos,command):
    if command == 'stay':
        outpos = pos
    elif command == 'up':
        outpos = (pos[0],pos[1]+1)
    elif command == 'down':
        outpos = (pos[0],pos[1]-1)
    elif command == 'left':
        outpos = (pos[0]-1,pos[1])
    elif command == 'right':
        outpos = (pos[0]+1,pos[1])
    else:
        print 'unknown command'
        raise
    return outpos

start_pos = (142,328)
end_pos = (199,371)
weather_matrix = train[1,:,10,:,:]
start_time = datetime.datetime(2017,1,1,hour=3,minute=0)
current_time = start_time
current_pos = start_pos
state_dict = {'stay':0,'up':0,'down':0,'left':0,'right':0}

# greedy policy, choose move that minimize the manhatten_distance, if cant make it, then wait
while True:
    if weather_matrix[start_time.hour-3,start_pos[0]-1,start_pos[1]-1] >= 15:
        print 'can not start, just crush'
        break
    else:
        print 'current pos is now at %s'%(current_pos,)
        print 'current time is now at %s'%(current_time)
        # update the dict using the manhatten_distance
        for k in state_dict.keys():
            state_dict[k] = manhatten_distance(end_pos,move(current_pos,k))
        # sorted the dict using the manhatten_distance
        sorted_dict = sorted(state_dict.items(), key=lambda x: x[1])

        # check weather is good
        if weather_matrix[current_time.hour-3,move(current_pos,sorted_dict[0][0])[0]-1,move(current_pos,sorted_dict[0][0])[1]-1] < 15:
            current_pos = move(current_pos,sorted_dict[0][0])
            print 'action %s is executed!'%(sorted_dict[0][0])
            print 'current pos is now at %s'%(current_pos,)

        elif weather_matrix[current_time.hour-3,move(current_pos,sorted_dict[1][0])[0]-1,move(current_pos,sorted_dict[1][0])[1]-1] < 15:
            current_pos = move(current_pos,sorted_dict[1][0])
            print 'action %s is executed!'%(sorted_dict[1][0])
            print 'current pos is now at %s'%(current_pos,)

        else:
            current_pos = move(current_pos,sorted_dict[2][0])
            print 'action %s is executed!'%(sorted_dict[2][0])
            print 'current pos is now at %s'%(current_pos,)

        # check weather the balloon is at the end pos
        if current_pos == end_pos:
            print 'Successfully arrived at end_pos %s'%(end_pos,)
            print 'total time consumed %s'%(current_time-start_time)
            break
        current_time += datetime.timedelta(minutes=2)