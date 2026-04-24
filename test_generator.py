from data_loader import batch_generator

# Simulate a data set of 10 items, process in batches of 3
dataset = list(range(10))

print("Procesing in batches:")
for batch_num, batch in enumerate(batch_generator(dataset, batch_size=3)):
    print(f"  Batch {batch_num + 1}: {batch}")

# Key insight: a generator is lazy — it only computes the next item when asked
gen = batch_generator(dataset, batch_size=3)
print()
print("Lazy evaluation:")
print("Next batch:", next(gen))
print("Next batch:", next(gen))
print("Next batch:", next(gen))