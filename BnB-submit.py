import numpy as np
import random
import sys
import time
from queue import PriorityQueue

EPS=1e-6

def float_equal(a,b):
    return abs(a-b)<EPS

def float_greater(a,b):
    return a>b+EPS

def float_less(a,b):
    return a<b-EPS

class Answer:
    def __init__(self, path=[], distance=0.0):
        self.path = path
        self.distance = distance
    def __str__(self):
        return f"Distance: {self.distance:.4f} Path: {self.path} "
    def __eq__(self, other):
        if not self.dist_eqal(other) or len(self.path)!=len(other.path):
            return False
        if len(self.path)==0:
            return True
        pos0=other.path.index(self.path[0])
        pos0_rev=other.path[::-1].index(self.path[0])
        return (self.path == other.path[pos0:]+other.path[:pos0])or(self.path == other.path[::-1][pos0_rev:]+other.path[::-1][:pos0_rev])
    def empty(self):
        return self.path == []
    def last_node(self):
        return self.path[-1]
    def first_node(self):
        return self.path[0]
    def add_node_copy(self, node, distance):
        return Answer(self.path+[node], self.distance+distance)
    def dist_eqal(self,other):
        return abs(self.distance-other.distance)<EPS

def get_shortest_path_step(n, origin_dist):
    # dist[i][step][j]
    dist=np.full((n, n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if(i!=j):
                dist[i][1][j]=origin_dist[i][j]
    for step in range(2,n):
        for i in range(n):
            for k in range(n):
                for j in range(n):
                    if dist[i][step][j] > dist[i][step-1][k] + dist[k][1][j]:
                        dist[i][step][j] = dist[i][step-1][k] + dist[k][1][j]
    return dist

def heuristic_sps(end_node, node_set, shortest_path_step):
    return shortest_path_step[end_node][len(node_set)]

class edge:
    def __init__(self, start, end, weight):
        self.start=start
        self.end=end
        self.weight=weight
    def __lt__(self, other):
        return self.weight<other.weight
    def __str__(self):
        return f"({self.start},{self.end},{self.weight})"

def get_edge_sorted_full(n, dist):
    edge_list=[]
    for i in range(n):
        for j in range(i+1,n):
            edge_list.append(edge(i,j,dist[i][j]))
    edge_list.sort()
    return edge_list

class disjoint_set:
    def __init__(self, n):
        self.father=[i for i in range(n)]
        self.rank=[0 for i in range(n)]
    def find(self, x):
        if self.father[x]!=x:
            self.father[x]=self.find(self.father[x])
        return self.father[x]
    def union(self, x, y):
        x=self.find(x)
        y=self.find(y)
        if x==y:
            return
        if self.rank[x]<self.rank[y]:
            self.father[x]=y
        else:
            self.father[y]=x
            if self.rank[x]==self.rank[y]:
                self.rank[x]+=1

def heuristic_mst(end_node, node_set, dist, edge_sorted_full):
    heuristic_list=np.zeros(dist.shape[0])
    set_num=len(node_set)
    mst_value=0.0
    djs=disjoint_set(dist.shape[0])
    for e in edge_sorted_full:
        if e.start in node_set and e.end in node_set:
            if djs.find(e.start) != djs.find(e.end):
                mst_value+=e.weight
                djs.union(e.start, e.end)
                set_num-=1
                if set_num==1:
                    break

    min_dist1=np.inf
    min_dist2=np.inf
    min_node1=-1
    for node in node_set:
        if dist[end_node][node]<min_dist1:
            min_dist2=min_dist1
            min_dist1=dist[end_node][node]
            min_node1=node
        elif dist[end_node][node]<min_dist2:
            min_dist2=dist[end_node][node]

    for node in node_set:
        heuristic_list[node]=mst_value+min_dist1
    heuristic_list[min_node1]+=min_dist2-min_dist1
    return heuristic_list

def get_min_edge_list(n,dist):
    min_edge_list_1plus2=np.zeros(n)
    min_edge_list_1=np.zeros(n)
    min_edge_list_2=np.zeros(n)
    for i in range(n):
        min_edge_1=min_edge_2=np.inf
        for j in range(n):
            if i!=j:
                if dist[i][j]<min_edge_1:
                    min_edge_2=min_edge_1
                    min_edge_1=dist[i][j]
                elif dist[i][j]<min_edge_2:
                    min_edge_2=dist[i][j]
        min_edge_list_1plus2[i]=(min_edge_1+min_edge_2)
        min_edge_list_1[i]=min_edge_1
        min_edge_list_2[i]=min_edge_2
    return min_edge_list_1, min_edge_list_2, min_edge_list_1plus2

def heuristic_min_edge_subset(end_node, node_set, half_dist):
    assert(len(node_set)>1)
    heuristic_list=np.zeros(half_dist.shape[0])
    min_edge_list_2=np.zeros(half_dist.shape[0])
    edge_sum=0.0
    node_set_with_end=node_set.copy()
    node_set_with_end.add(end_node)
    for s_node in node_set_with_end:
        min_edge1=min_edge2=np.inf
        for e_node in node_set_with_end:
            if s_node!=e_node:
                if half_dist[s_node][e_node]<min_edge1:
                    min_edge2=min_edge1
                    min_edge1=half_dist[s_node][e_node]
                elif half_dist[s_node][e_node]<min_edge2:
                    min_edge2=half_dist[s_node][e_node]
        min_edge_list_2[s_node]=min_edge2
        edge_sum+=min_edge1+min_edge2
    edge_sum-=min_edge_list_2[end_node]
    for node in node_set:
        heuristic_list[node]=edge_sum-min_edge_list_2[node]
    return heuristic_list

def heuristic_cmp_mst_edgesub(end_node, node_set, dist, edge_sorted_full, half_dist):
    global a,b
    val1=heuristic_mst(end_node, node_set, dist, edge_sorted_full)
    val2=heuristic_min_edge_subset(end_node, node_set, half_dist)
    heuristic_list=np.zeros(dist.shape[0])
    for node in node_set:
        if float_greater(val1[node],val2[node]):
            a+=1
            heuristic_list[node]=val1[node]
        else:
            b+=1
            heuristic_list[node]=val2[node]
    return heuristic_list

def bnb_dfs(cur_ans, best_ans, node_set, dist, heuristic):
    global h_count
    if len(node_set)== 1:
        node=node_set.pop()
        actual_dist=dist[node][cur_ans.last_node()]+dist[node][cur_ans.first_node()]+cur_ans.distance
        if actual_dist<best_ans.distance:
            best_ans=Answer(cur_ans.path+[node],actual_dist)
        return best_ans
    global h_time
    t1=time.time()
    heuristic_list=heuristic(cur_ans.first_node(), node_set)
    t2=time.time()
    h_time+=t2-t1
    h_count+=len(node_set)
    for node in node_set:
        lower_bound=heuristic_list[node]+dist[cur_ans.last_node()][node]+cur_ans.distance
        if(lower_bound>=best_ans.distance):
            continue
        best_ans=bnb_dfs(cur_ans.add_node_copy(node,dist[cur_ans.last_node()][node]),best_ans,node_set-{node},dist,heuristic)
    return best_ans

def gen_init_ans_true(n, dist):
    init_ans=Answer(list(range(n)),0)
    random.shuffle(init_ans.path)
    for i in range(n):
        init_ans.distance+=dist[init_ans.path[i]][init_ans.path[(i+1)%n]]
    return init_ans

def gen_init_ans(n,dist):
    init_ans=Answer([],distance=np.inf)
    for node in range(n):
        ans=Answer([node])
        node_set=set(range(n))-{node}
        while(len(node_set)>2):
            min_dist=min_dist2=np.inf
            min_node=min_node2=-1
            for node in node_set:
                if dist[ans.last_node()][node]<min_dist:
                    min_dist=dist[ans.last_node()][node]
                    min_node=node
                if dist[ans.first_node()][node]<min_dist2:
                    min_dist2=dist[ans.first_node()][node]
                    min_node2=node
            if min_dist2<min_dist:
                ans=Answer([min_node2]+ans.path,ans.distance+min_dist2)
                node_set.remove(min_node2)
            else:
                ans=Answer(ans.path+[min_node],ans.distance+min_dist)
                node_set.remove(min_node)

        if(len(node_set)==2):
            node_1=node_set.pop()
            node_2=node_set.pop()
            if dist[ans.last_node()][node_1]+dist[node_2][ans.first_node()]<=dist[ans.last_node()][node_2]+dist[node_1][ans.first_node()]:
                ans.distance+=dist[ans.last_node()][node_1]+dist[node_2][ans.first_node()]+dist[node_1][node_2]
                ans.path+=[node_1,node_2]
            else:
                ans.distance+=dist[ans.last_node()][node_2]+dist[node_1][ans.first_node()]+dist[node_1][node_2]
                ans.path+=[node_2,node_1]
        elif(len(node_set)==1):
            node=node_set.pop()
            ans.path.append(node)
            ans.distance+=dist[ans.last_node()][node]+dist[node][ans.first_node()]
        if ans.distance<init_ans.distance:
            init_ans=ans
    return init_ans

def get_array_from_file(file):
    n = int(np.loadtxt(file, max_rows=1))
    dist = np.loadtxt(file, skiprows=1)
    return n, dist

def gen_init_ans_from_file(file):
    n = int(np.loadtxt(file, max_rows=1))
    dist = np.loadtxt(file, skiprows=1)
    return gen_init_ans(n,dist)

def solution(n,dist,heuristic_type,init_ans=None, print_info=False):
    global h_time, h_count, a, b
    h_time=0.0
    h_count=0
    a=b=0
    best_ans=gen_init_ans(n,dist) if init_ans is None else init_ans
    t1=time.time()
    if heuristic_type=="trivial":
        zeros=[0 for _ in range(n)]
        heuristic=lambda end_node, node_set: zeros
    elif heuristic_type=="sps":
        shortest_path_step=get_shortest_path_step(n,dist)
        heuristic=lambda end_node, node_set: heuristic_sps(end_node, node_set, shortest_path_step)
    elif heuristic_type=="mst":
        edge_sorted_full=get_edge_sorted_full(n,dist)
        heuristic=lambda end_node, node_set: heuristic_mst(end_node, node_set, dist, edge_sorted_full)
    elif heuristic_type=="edge":
        heuristic=lambda end_node, node_set: heuristic_min_edge_subset(end_node, node_set, dist/2)
    elif heuristic_type=="mst_ed":
        edge_sorted_full=get_edge_sorted_full(n,dist)
        heuristic=lambda end_node, node_set: heuristic_cmp_mst_edgesub(end_node, node_set, dist, edge_sorted_full, dist/2)
    else:
        return Answer()
    t2=time.time()
    if print_info:
        print(f"Preprocessing time: {t2-t1:.6f} s")
    start_node=0
    best_ans=bnb_dfs(Answer([start_node]),best_ans,set(range(n))-{start_node},dist,heuristic)
    if print_info and h_count!=0:
        print(f"Heuristic time: {h_time:.6f} s, {h_time/h_count:.10f} s, {h_count} times")
    if print_info and (a>0 or b>0):
        print(f"{a} {b}")
    return best_ans

def tsp_from_file(file,heuristic_type,init_ans=None):
    n, dist = get_array_from_file(file)
    return solution(n,dist,heuristic_type,init_ans)

# Usage: python3 BnB-submit.py <file> <heuristic_type>
# heuristic_type: trivial, sps, mst, edge, mst_ed
if __name__ == "__main__":
    file=sys.argv[1]
    heuristic_type=sys.argv[2] if(len(sys.argv)>2) else "trivial"
    print(tsp_from_file(file,heuristic_type))
