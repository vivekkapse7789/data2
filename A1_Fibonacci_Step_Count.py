# 1. Iterative (Non-Recursive) Fibonacci
def fibonacci_iterative(n):
    steps = 0
    a, b = 0, 1
    steps += 1                      # initial assignment
    if n == 0:
        steps += 1                  # comparison
        return 0, steps
    for _ in range(2, n + 1):
        steps += 1                  # loop check
        a, b = b, a + b
        steps += 1                  # addition + assignment
    steps += 1                      # return
    return b, steps

n = 10
result, steps = fibonacci_iterative(n)
print(f"Fibonacci({n}) using iteration = {result}")
print(f"Step count (iterative)       = {steps}")


# 2. Recursive Fibonacci
recursive_steps = 0

def fibonacci_recursive(n):
    global recursive_steps
    recursive_steps += 1            # one step per function call
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

recursive_steps = 0
result = fibonacci_recursive(n)
print(f"Fibonacci({n}) using recursion = {result}")
print(f"Step count (recursive)        = {recursive_steps}")


# 3. Printing the Fibonacci Series
def fibonacci_series(n):
    series = [0, 1]
    for i in range(2, n):
        series.append(series[-1] + series[-2])
    return series[:n]

print("First 15 Fibonacci numbers:")
print(fibonacci_series(15))


# 4. Comparison of Step Counts for Different n
import time
print(f"{'n':>4} | {'Fib(n)':>8} | {'Iter steps':>10} | {'Rec steps':>10} | {'Iter time (s)':>14} | {'Rec time (s)':>13}")
print("-" * 75)
for n in [5, 10, 15, 20, 25]:
    t0 = time.perf_counter(); val, it_steps = fibonacci_iterative(n); t1 = time.perf_counter()
    recursive_steps = 0
    t2 = time.perf_counter(); fibonacci_recursive(n); t3 = time.perf_counter()
    print(f"{n:>4} | {val:>8} | {it_steps:>10} | {recursive_steps:>10} | {t1-t0:>14.7f} | {t3-t2:>13.7f}")


# 5. Plot
import matplotlib.pyplot as plt
ns = list(range(1, 26))
it_counts, rec_counts = [], []
for k in ns:
    it_counts.append(fibonacci_iterative(k)[1])
    recursive_steps = 0
    fibonacci_recursive(k)
    rec_counts.append(recursive_steps)

plt.figure(figsize=(8, 4.5))
plt.plot(ns, it_counts, marker='o', label='Iterative O(n)')
plt.plot(ns, rec_counts, marker='s', label='Recursive O(2^n)')
plt.yscale('log')
plt.xlabel('n'); plt.ylabel('Step count (log scale)')
plt.title('Step Count: Iterative vs Recursive Fibonacci')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
