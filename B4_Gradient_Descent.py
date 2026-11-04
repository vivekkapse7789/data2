import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x + 3) ** 2

def df(x):
    return 2 * (x + 3)


def gradient_descent(start_x, learning_rate=0.1, precision=1e-6, max_iters=10000):
    x = start_x
    history = [x]
    for i in range(max_iters):
        prev_x = x
        x = x - learning_rate * df(x)
        history.append(x)
        if abs(x - prev_x) < precision:
            break
    return x, i + 1, history


x_min, iters, path = gradient_descent(start_x=2, learning_rate=0.1)
print(f"Local minimum occurs at x = {x_min:.6f}")
print(f"Minimum value of y      = {f(x_min):.10f}")
print(f"Number of iterations    = {iters}")


print(f"{'Iter':>4} | {'x':>10} | {'y = (x+3)^2':>12} | {'gradient':>10}")
print("-" * 46)
for i, xv in enumerate(path[:11]):
    print(f"{i:>4} | {xv:>10.5f} | {f(xv):>12.5f} | {df(xv):>10.5f}")
print("  ...")


xs = np.linspace(-8, 4, 400)
plt.figure(figsize=(8, 5))
plt.plot(xs, f(xs), label='y = (x + 3)^2')
plt.scatter(path, [f(p) for p in path], color='red', s=20, zorder=3, label='Gradient descent steps')
plt.plot(path, [f(p) for p in path], 'r--', alpha=0.5)
plt.scatter([x_min], [f(x_min)], color='green', s=80, zorder=4, label=f'Minimum ≈ ({x_min:.3f}, {f(x_min):.3f})')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Gradient Descent on y = (x + 3)^2 starting at x = 2')
plt.legend(); plt.grid(alpha=0.3); plt.show()


# Effect of Learning Rate
for lr in [0.01, 0.05, 0.1, 0.5, 0.9]:
    xm, it, _ = gradient_descent(2, learning_rate=lr)
    print(f"learning rate = {lr:<5} -> x_min = {xm:.6f}, iterations = {it}")
