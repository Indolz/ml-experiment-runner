def count_up(limit: int):
    i = 0 
    while i < limit:
        print(f" About to yield {i}")
        yield i
        print(f" Resume after yielding {i}")
        i += 1
    print(" Generator exhausted")

# Creating the generator does NOT run any code yet
gen = count_up(3)
print("Generator created - nothing has run yet")
print()

# Each next() runs until the next yield, then pauses
print("First next():")
val = next(gen)
print(f"Got back: {val}")
print()

# for loop calls next() automatically until exghausted
print("For loop takes the rest:")
for val in gen:
    print(f"Got back: {val}")
print()

# Real-world use - memory-efficient batching
def batch_generator(data: list, batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

dataset = list(range(10))
print("Dataset in batches of 3:")
for num, batch in enumerate(batch_generator(dataset, 3)):
    print(f" Batch {num + 1}: {batch}")