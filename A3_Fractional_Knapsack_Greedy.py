class Item:
    def __init__(self, name, value, weight):
        self.name = name
        self.value = value
        self.weight = weight
        self.ratio = value / weight


def fractional_knapsack(capacity, items):
    items = sorted(items, key=lambda i: i.ratio, reverse=True)
    total_value = 0.0
    remaining = capacity

    print(f"{'Item':<6}{'Value':<8}{'Weight':<8}{'Ratio':<8}{'Fraction taken':<16}{'Value added'}")
    print("-" * 60)
    for it in items:
        if remaining == 0:
            fraction = 0
        elif it.weight <= remaining:
            fraction = 1
        else:
            fraction = remaining / it.weight
        added = it.value * fraction
        remaining -= it.weight * fraction
        total_value += added
        print(f"{it.name:<6}{it.value:<8}{it.weight:<8}{it.ratio:<8.2f}{fraction:<16.2f}{added:.2f}")
    return total_value


# Example 1
items = [Item('I1', 60, 10), Item('I2', 100, 20), Item('I3', 120, 30)]
W = 50
print(f"Knapsack capacity = {W}\n")
max_val = fractional_knapsack(W, items)
print(f"\nMaximum value in knapsack = {max_val:.2f}")


# Example 2
items2 = [Item('A', 10, 2), Item('B', 5, 3), Item('C', 15, 5),
          Item('D', 7, 7), Item('E', 6, 1), Item('F', 18, 4), Item('G', 3, 1)]
W2 = 15
print(f"\nKnapsack capacity = {W2}\n")
max_val = fractional_knapsack(W2, items2)
print(f"\nMaximum value in knapsack = {max_val:.2f}")
