from experiment import MLExperiment, RegularisedExperiment

base = MLExperiment('baseline', 0.01)
reg = RegularisedExperiment('Regularised', 0.01,
                            lambda_reg=0.01)

base.run(4)
reg.run(4)

print('Base:')
print(base)
print(base.results)

print()
print('Regularised:')
print(reg)
print(reg.results)

print()
print(isinstance(reg, MLExperiment))
print(isinstance(reg, RegularisedExperiment))
print(isinstance(base, RegularisedExperiment))

print()
print("--- Timing test ---")
big_exp = MLExperiment('Timing-test', 0.001)
big_exp.run(1000000)
print(f"Ran {len(big_exp)} epchos")