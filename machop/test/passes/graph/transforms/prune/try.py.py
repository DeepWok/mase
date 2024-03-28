import torch
from collections import Counter
from heapq import heappush, heappop, heapify

f = open("/mnt/d/imperial/second_term/adls/projects/mase/all_weights.txt","r")   
weight_list = f.read()    
f.close()  

frequency = Counter(weight_list)

print(frequency)
'''
heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
heapify(heap)
while len(heap) > 1:
    lo = heappop(heap)
    hi = heappop(heap)
    for pair in lo[1:]:
        pair[1] = '0' + pair[1]
    for pair in hi[1:]:
        pair[1] = '1' + pair[1]
    heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
huffman_tree = sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
'''