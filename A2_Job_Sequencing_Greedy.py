class Job:
    def __init__(self, job_id, deadline, profit):
        self.id = job_id
        self.deadline = deadline
        self.profit = profit

    def __repr__(self):
        return f"{self.id}(d={self.deadline}, p={self.profit})"


def job_sequencing(jobs):
    # Step 1: sort by profit (descending)
    jobs = sorted(jobs, key=lambda j: j.profit, reverse=True)
    print("Jobs sorted by profit:", jobs, "\n")

    max_deadline = max(j.deadline for j in jobs)
    slots = [None] * max_deadline          # slot i represents time (i, i+1]
    total_profit = 0

    # Step 2: place each job in the latest free slot before its deadline
    for job in jobs:
        for s in range(min(max_deadline, job.deadline) - 1, -1, -1):
            if slots[s] is None:
                slots[s] = job
                total_profit += job.profit
                print(f"Job {job.id} scheduled in slot {s+1}")
                break
        else:
            print(f"Job {job.id} rejected (no free slot before deadline {job.deadline})")

    return slots, total_profit


# Example 1
jobs = [Job('J1', 2, 100), Job('J2', 1, 19), Job('J3', 2, 27),
        Job('J4', 1, 25), Job('J5', 3, 15)]

print(f"{'Job':<5}{'Deadline':<10}{'Profit':<8}")
for j in jobs:
    print(f"{j.id:<5}{j.deadline:<10}{j.profit:<8}")
print()

slots, profit = job_sequencing(jobs)
sequence = [j.id for j in slots if j is not None]
print("\nOptimal job sequence :", " -> ".join(sequence))
print("Maximum profit       :", profit)


# Example 2
jobs2 = [Job('a', 4, 20), Job('b', 1, 10), Job('c', 1, 40),
         Job('d', 1, 30), Job('e', 3, 50), Job('f', 2, 35)]

slots, profit = job_sequencing(jobs2)
sequence = [j.id for j in slots if j is not None]
print("\nOptimal job sequence :", " -> ".join(sequence))
print("Maximum profit       :", profit)
