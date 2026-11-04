# 1. Dynamic Programming Solution
def knapsack_dp(W, wt, val):
    n = len(val)
    K = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            if wt[i-1] <= w:
                K[i][w] = max(K[i-1][w], val[i-1] + K[i-1][w - wt[i-1]])
            else:
                K[i][w] = K[i-1][w]

    # Trace back the chosen items
    chosen, w = [], W
    for i in range(n, 0, -1):
        if K[i][w] != K[i-1][w]:
            chosen.append(i - 1)
            w -= wt[i-1]
    return K[n][W], sorted(chosen), K


val = [60, 100, 120]
wt  = [10, 20, 30]
W   = 50

best, chosen, table = knapsack_dp(W, wt, val)
print("Values  :", val)
print("Weights :", wt)
print("Capacity:", W)
print("\nMaximum value (DP)  :", best)
print("Items selected      :", [f"Item{i+1}(w={wt[i]}, v={val[i]})" for i in chosen])
print("Total weight used   :", sum(wt[i] for i in chosen))


# DP table for a small instance
val_s = [1, 4, 5, 7]
wt_s  = [1, 3, 4, 5]
W_s   = 7
best_s, chosen_s, K = knapsack_dp(W_s, wt_s, val_s)

header = " i\\w | " + " ".join(f"{w:>3}" for w in range(W_s + 1))
print(header); print("-" * len(header))
for i, row in enumerate(K):
    print(f"{i:>4} | " + " ".join(f"{x:>3}" for x in row))
print("\nMaximum value:", best_s, "| Items selected:", [i+1 for i in chosen_s])


# 2. Branch and Bound Solution
from queue import PriorityQueue

class Node:
    def __init__(self, level, profit, weight, bound, taken):
        self.level, self.profit, self.weight, self.bound, self.taken = level, profit, weight, bound, taken
    def __lt__(self, other):               # max-heap on bound
        return self.bound > other.bound

def bound(node, n, W, items):
    if node.weight >= W:
        return 0
    b, tw, j = node.profit, node.weight, node.level + 1
    while j < n and tw + items[j][1] <= W:
        tw += items[j][1]; b += items[j][0]; j += 1
    if j < n:
        b += (W - tw) * items[j][0] / items[j][1]
    return b

def knapsack_bb(W, wt, val):
    n = len(val)
    order = sorted(range(n), key=lambda i: val[i] / wt[i], reverse=True)
    items = [(val[i], wt[i]) for i in order]
    pq = PriorityQueue()
    root = Node(-1, 0, 0, 0, [])
    root.bound = bound(root, n, W, items)
    pq.put(root)
    best_profit, best_taken, explored = 0, [], 0

    while not pq.empty():
        u = pq.get(); explored += 1
        if u.bound <= best_profit or u.level == n - 1:
            continue
        lvl = u.level + 1
        # include next item
        v = Node(lvl, u.profit + items[lvl][0], u.weight + items[lvl][1], 0, u.taken + [order[lvl]])
        if v.weight <= W and v.profit > best_profit:
            best_profit, best_taken = v.profit, v.taken
        v.bound = bound(v, n, W, items)
        if v.bound > best_profit:
            pq.put(v)
        # exclude next item
        x = Node(lvl, u.profit, u.weight, 0, u.taken)
        x.bound = bound(x, n, W, items)
        if x.bound > best_profit:
            pq.put(x)
    return best_profit, sorted(best_taken), explored


best_bb, taken_bb, nodes = knapsack_bb(W, wt, val)
print("Maximum value (Branch & Bound):", best_bb)
print("Items selected                :", [i+1 for i in taken_bb])
print("Nodes explored                :", nodes)

best_bb2, taken_bb2, nodes2 = knapsack_bb(W_s, wt_s, val_s)
print("\nSmall instance -> max value:", best_bb2, "| items:", [i+1 for i in taken_bb2], "| nodes explored:", nodes2)
