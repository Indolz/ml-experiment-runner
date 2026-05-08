class Car:
    
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    def description(self):
        return f"{self.year} {self.make} {self.model}"
    
class ElectricCar(Car):
    
    def __init__(self, make, model, year, battery_size):
        super().__init__(make, model, year)
        self.battery_size = battery_size

    def describe_battery(self):
        print(f"Battery:{self.battery_size} kwh")

my_car = Car("Toyota", "Corolla", 2020)
print(my_car.description())

my_tesla = ElectricCar("Tesla", "Model 3", 2023, 75)
print(my_tesla.description())   # inherited from Car
my_tesla.describe_battery()     # defined in ElectricCar
