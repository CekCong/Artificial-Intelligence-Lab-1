#Part 1 Finding a Shortest Path with An Energy Budget
import json

with open('Coord.json', 'r') as Coord:
    Coord =  json.load(Coord)

with open('Cost.json', 'r') as Cost:
    Cost =  json.load(Cost)

with open('Dist.json', 'r') as Dist:
    Dist =  json.load(Dist)

with open('G.json', 'r') as G:
    G =  json.load(G)


#Task 1: Use Uniform Cost Search
import heapq
pq = []
visited = []
heapq.heappush(pq,(0,"1",0, "1")) #dist, node, energy, path
while pq:
    dist, node, energy, path = heapq.heappop(pq)
    if node in visited:
        continue
    visited.append(node)
    if node == "50":
        break
    for neighbour in G[node]:
        if neighbour in visited:  #skip visited neighbours
            continue
        new_dist = dist + Dist[f"{node},{neighbour}"]
        new_energy = energy + Cost[f"{node},{neighbour}"]
        newpath = path + "->" + neighbour
        heapq.heappush(pq,(new_dist, neighbour, new_energy,newpath))       
print("Part 1 Finding a Shortest Path with An Energy Budget --------------------------------")        
print("Task 1: --------------------------------")
print(f"""Shortest path: {path} 
Shortest distance: {dist}. 
Total energy cost: {energy}. """)

#Task 2:  Uninformed Search Algorithm
pq = []
budget = 287932
heapq.heappush(pq,(0,"1",0, "1")) #dist, node, energy, path
visited = {"1": (0,0)}  #node: (dist, energy)
while pq:
    dist, node, energy, path = heapq.heappop(pq)
    if node == "50":
        break
    for neighbour in G[node]:
        new_dist = dist + Dist[f"{node},{neighbour}"]
        new_energy = energy + Cost[f"{node},{neighbour}"]
        newpath = path + "->" + neighbour
        if new_energy > budget: #skip paths that exceed energy budget
            continue
        elif neighbour not in visited: #not visited yet
            heapq.heappush(pq,(new_dist, neighbour, new_energy,newpath)) 
            visited[neighbour] = (new_dist, new_energy)    
        elif new_dist >= visited[neighbour][0] and new_energy >= visited[neighbour][1]: #longer dist and more energy
            continue
        else: #shorter dist or less energy
            heapq.heappush(pq,(new_dist, neighbour, new_energy,newpath)) 
            visited[neighbour] = (new_dist, new_energy)  
print("\nTask 2: --------------------------------")
print(f"""Shortest path: {path} 
Shortest distance: {dist}. 
Total energy cost: {energy}. """)


#Task 3: A* Search Algorithm
import math
H = {}
for node in G:
    H[node] = math.sqrt((Coord[node][0]-Coord["50"][0])**2 + (Coord[node][1]-Coord["50"][1])**2) 
pq = []
budget = 287932
heapq.heappush(pq,(0,0,"1",0, "1")) # f(n),dist, node, energy, path
visited = {"1": (0,0)}  #node: [dist, energy]
while pq:
    f_n, dist, node, energy, path = heapq.heappop(pq)
    if node == "50":
        break
    for neighbour in G[node]:
        new_dist = dist + Dist[f"{node},{neighbour}"]
        new_energy = energy + Cost[f"{node},{neighbour}"]
        newpath = path + "->" + neighbour
        fn_new = new_dist + H[neighbour]
        if new_energy > budget: #skip paths that exceed energy budget
            continue
        elif neighbour not in visited: #not visited yet
            heapq.heappush(pq,(fn_new, new_dist, neighbour, new_energy, newpath))
            visited[neighbour] = (new_dist, new_energy)    
        elif new_dist >= visited[neighbour][0] and new_energy >= visited[neighbour][1]: #longer dist and more energy
            continue
        else: #shorter dist or less energy
            heapq.heappush(pq,(fn_new, new_dist, neighbour, new_energy,newpath)) 
            visited[neighbour] = (new_dist, new_energy)  
print("\nTask 3: --------------------------------")
print(f"""Shortest path: {path} 
Shortest distance: {dist}. 
Total energy cost: {energy}. \n""")


#Part 2 Solving MDP and Reinforcement Learning Problems Using a Grid World
#Task 1:
#initialize variables
start = (0,0)
goal =  (4,4)
blocked = [(2,1),(2,3)]
step_cost = -1
width = 5
actions = {
    'Up': [(1,0), 'Left', 'Right'], #up, left, right
    'Down': [(-1,0), 'Left', 'Right'], #down, left, right
    'Left': [(0,-1), 'Up', 'Down'], #left, up, down
    'Right': [(0,1), 'Up', 'Down'] #right, up, down
}
import random
# Task 1.1: Value Iteration
# matrix for value function V(s)
V = [[0 for _ in range(width)] for _ in range(width)] #value function V(s)
#dictionary for Q table
Q= {}
for x in range(width):
    for y in range(width):
        Q[(x,y)] = {
            'Up': 0,     #represent Q(s,a) for UP action
            'Down': 0,
            'Left': 0,
            'Right': 0
        }
#matrix for policy
policy = [[None for _ in range(width)] for _ in range(width)] 

def Q_ip1 (state,action):  #find Q(s,a) for iteration i+1
    if state == goal or state in blocked:
        return 0
    #Get all possible next state
    possible_new_state = []
    new_state = (state[0]+actions[action][0][0], state[1]+actions[action][0][1]) 
    possible_new_state.append(new_state)
    actual_action = actions[action][1]
    new_state = (state[0]+actions[actual_action][0][0], state[1]+actions[actual_action][0][1])
    possible_new_state.append(new_state)
    actual_action = actions[action][2]
    new_state = (state[0]+actions[actual_action][0][0], state[1]+actions[actual_action][0][1])
    possible_new_state.append(new_state)
    #if hit a wall or blocked cell, then stay in the same state
    reward_sum = 0
    for i in range(len(possible_new_state)):
        new_state = possible_new_state[i]
        if new_state in blocked or new_state[0] < 0 or new_state[0] >= width or new_state[1] < 0 or new_state[1] >= width: 
            new_state = state
        if new_state == goal:
            reward = 10
        else:
            reward = step_cost
        if i == 0:
            reward_sum = reward_sum + 0.8*(reward + 0.9*V[new_state[0]][new_state[1]] )#reward + step cost + gamma * V(s')
        else:
            reward_sum = reward_sum + 0.1*(reward + 0.9*V[new_state[0]][new_state[1]] )#reward + step cost + gamma * V(s')
    return reward_sum

def V_ip2(state): #find V(s) for iteration i+1
    #if state == goal:
    #    return 0
    return max(Q_ip1(state, action) for action in actions)

def value_iteration(): # value iteration
    omega = 0.001 #small value
    while True:
        theta = 0
        for x in range(width):
            for y in range(width):
                for action in actions:
                    old_Q = Q[(x,y)][action]
                    Q[(x,y)][action] = Q_ip1((x,y) , action) #update Q table
                    theta =  max(theta, abs(old_Q-Q[(x,y)][action]))
                #update V table
                V[x][y] = V_ip2((x,y))
        #Check if converges
        if theta < omega:
            break
    #update policy after Nth iterations
    for x in range(width):
        for y in range(width):
            state = (x,y)
            if state == goal or state in blocked:
                continue   #skip blocked and goal state
            best_action = None
            best_value = float('-inf')
            for action in actions:
                if Q[state][action] > best_value:
                    best_action = action
                    best_value = Q[state][action]
            policy[x][y] = best_action

import time
start_t = time.time()
value_iteration()
end = time.time()
print("Part 2 Solving MDP and Reinforcement Learning Problems Using a Grid World  --------------------------------")        
print("Task 1: --------------------------------")
print("V Table for value iteration: ")
for i in range(4,-1,-1):
    print(V[i])
print("\n")
print("Policy for value iteration: ")
for i in range(4,-1,-1):
    print(policy[i])
print(f"Time taken: {end - start_t:.4f} seconds\n")


# Task 1.2: Policy Iteration
# matrix for value function V(s)
V = [[0 for _ in range(width)] for _ in range(width)] #value function V(s)
#initialise policy
policy = [["Up", "Down", "Left", "Right","Up"],
          ["Up", "Down", "Left", "Right","Up"],
          ["Up", None, "Left", None, "Left"],
          ["Up", "Down", "Left", "Right","Up"],
          ["Up", "Down", "Left", "Right",None]
          ] 

def policy_iteration(): # policy iteration 
    global policy
    while True:
        while True:
            #Policy Evaluation
            omega = 0.001 #small value
            theta = 0
            for x in range(width):
                for y in range(width):
                    state = (x,y)
                    if state == goal or state in blocked:
                        continue
                    old_v = V[x][y]
                    new_v = Q_ip1(state,policy[x][y])
                    diff_in_v = abs(old_v-new_v)
                    theta = max( theta,diff_in_v)
                    V[x][y] =  new_v #update V table
            if theta<omega: #break if converges
                break
        #Policy Improvement
        new_policy = [[None for _ in range(width)] for _ in range(width)]
        for x in range(width):
            for y in range(width):
                state = (x,y)
                if state == goal or state in blocked:
                    continue
                best_action = None
                best_value = float('-inf')
                for action in actions:
                    Q_value = Q_ip1(state,action)
                    if Q_value > best_value:
                        best_action = action
                        best_value = Q_value
                new_policy[x][y] = best_action
        if new_policy == policy:
            break #stop iteration if policy is stable
        # Update policy
        policy = [row[:] for row in new_policy]

start_t = time.time()
policy_iteration()
end = time.time()
print("V Table for policy iteration: ")
for i in range(4,-1,-1):
    print(V[i])
print("\n")
print("Policy for policy iteration: ")
for i in range(4,-1,-1):
    print(policy[i])
print(f"Time taken: {end - start_t:.4f} seconds\n")


#Task 2:
#dictionary for Q table
Q= {}
for x in range(width):
    for y in range(width):
        Q[(x,y)] = {
            'Up': 0,     #represent Q(s,a) for UP action
            'Down': 0,
            'Left': 0,
            'Right': 0
        }
#matrix for policy
#initialise policy
policy = [["Up", "Up", "Up", "Up","Up"],
          ["Up", "Up", "Up", "Up","Up"],
          ["Up", None, "Up", None, "Up"],
          ["Up", "Up", "Up", "Up","Up"],
          ["Up", "Up", "Up", "Up",None]
          ] 

def generate_episode():
    curr = start
    rewards = [] #keep track of reward at each transition
    state_action = []#keep track of (state,action)
    count = 0
    while curr != goal and count <100: #prevent loop
        #Figure out what is the actual and new state for stachastic prob
        number = random.random() #Generate a random number from 0 to 1
        intended_action = policy[curr[0]][curr[1]]
        if number < 0.8:
            actual_action = intended_action
            new_state = (curr[0]+actions[actual_action][0][0], curr[1]+actions[actual_action][0][1])
        elif number <0.9:
            actual_action = actions[intended_action][1]
            new_state = (curr[0]+actions[actual_action][0][0], curr[1]+actions[actual_action][0][1])
        else:
            actual_action = actions[intended_action][2]
            new_state = (curr[0]+actions[actual_action][0][0], curr[1]+actions[actual_action][0][1])
        #If not valid state then stay the same 
        if new_state in blocked or new_state[0] < 0 or new_state[0] >= width or new_state[1] < 0 or new_state[1] >= width: 
            new_state = curr
        #update reward
        if new_state == goal:
            rewards.append(10)
        else:
            rewards.append(-1)
        #update state action
        state_action.append((curr,intended_action))
        curr = new_state
        count+=1
    return state_action, rewards

#reward for each (s,a)
reward_dictionary= {}
for x in range(width):
    for y in range(width):
        for action in actions:
            reward_dictionary[(x,y),action] = []

def monte_carlo(): # monte carlo
    while True:
        state_action, rewards = generate_episode()
        omega = 0.001 #small value
        theta = 0
        #Policy Evaluation
        visited = set() #O(1) runtime)        
        for i in range(len(state_action)):
            #Calculate Return
            s_a = state_action[i]
            if s_a in visited:
                continue  #skip if not 1st occurrence
            total, count = 0,0
            for r in rewards[i:]:
                total = total + r*0.9**count  #discounted rewards
                count+=1
            reward_dictionary[s_a].append(total) 
            #Update Q(s,a)
            old_Q = Q[s_a[0]][s_a[1]]
            count = len(reward_dictionary[s_a])
            Q[s_a[0]][s_a[1]] = (old_Q*(count-1) + total)/count
            #Check if Q changes a lot
            theta =  max(theta, abs(old_Q-Q[s_a[0]][s_a[1]]))
            visited.add(s_a)
        #Policy Improvement if not yet converge
        for s_a in visited:
            state = s_a[0]
            best_action = None
            best_value = float('-inf')
            for action in actions:
                if Q[state][action] > best_value:
                    best_action = action
                    best_value = Q[state][action]
            #Update policy with some randomness
            number = random.random() #Generate a random number from 0 to 1
            x,y = state[0], state[1]
            if number < 0.9:
                policy[x][y] = best_action
            else:
                action_list = list(actions.keys())
                policy[x][y] = random.choice(action_list)
        #Check if converges
        if theta < omega:
            break

start_t = time.time()
monte_carlo()
end = time.time()
print("Task 2: --------------------------------")
print("Policy for Monte Carlo: ")
for i in range(4,-1,-1):
    print(policy[i])
print(f"Time taken: {end - start_t:.4f} seconds\n")


#Task 3:
#dictionary for Q table
Q= {}
for x in range(width):
    for y in range(width):
        Q[(x,y)] = {
            'Up': 0,     #represent Q(s,a) for UP action
            'Down': 0,
            'Left': 0,
            'Right': 0
        }
#matrix for policy
policy = [[None for _ in range(width)] for _ in range(width)]

def q_learning(): # q_learning
    omega = 0.1 #small value
    while True:
        theta = 0
        curr = start
        while curr != goal:
            #Get current optimal action
            best_action = None
            best_value = float('-inf')
            for action in actions:
                if Q[curr][action] > best_value:
                    best_action = action
                    best_value = Q[curr][action]
            #Choose current optimal action with 1-E probability, else random
            number = random.random() #Generate a random number from 0 to 1
            if number < 0.9:
                chosen_action = best_action
            else:
                action_list = list(actions.keys())
                chosen_action = random.choice(action_list)
            #Stochastic env
            number = random.random() #Generate a random number from 0 to 1
            if number < 0.8:
                actual_action = chosen_action
                new_state = (curr[0]+actions[actual_action][0][0], curr[1]+actions[actual_action][0][1])
            elif number <0.9:
                actual_action = actions[chosen_action][1]
                new_state = (curr[0]+actions[actual_action][0][0], curr[1]+actions[actual_action][0][1])
            else:
                actual_action = actions[chosen_action][2]
                new_state = (curr[0]+actions[actual_action][0][0], curr[1]+actions[actual_action][0][1])
            #If not valid state then stay the same 
            if new_state in blocked or new_state[0] < 0 or new_state[0] >= width or new_state[1] < 0 or new_state[1] >= width: 
                new_state = curr
            #update reward
            if new_state == goal:
                immediate_reward = 10
            else:
                immediate_reward = -1
            #Get next state optimal action
            next_state_best_value = max(Q[new_state].values())
            #Calculate Q_new 
            Q_old = Q[curr][chosen_action]
            Q[curr][chosen_action] = Q[curr][chosen_action] + 0.1*(immediate_reward + 0.9*next_state_best_value - Q[curr][chosen_action])
            theta =  max(theta, abs(Q_old - Q[curr][chosen_action]))
            #update state action
            curr = new_state
        if theta < omega:
            break
import time
start_t = time.time()
q_learning()
end = time.time()
# update policy
for state in Q:
    if state == goal or state in blocked:
        continue
    best_action = None
    best_value = float('-inf')
    for action in actions:
        if Q[state][action] > best_value:
            best_action = action
            best_value = Q[state][action]
    policy[state[0]][state[1]] = best_action
print("Task 3: --------------------------------")
print("Policy for Q-Learning: ")
for i in range(4,-1,-1):
    print(policy[i])
print(f"Time taken: {end - start_t:.4f} seconds\n")