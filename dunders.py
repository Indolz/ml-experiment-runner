class MLDataset:
    """A simple data set class that demostrates key dunders."""

    def __init__(self, name: str, data: list):
        self.name = name
        self.data = data

    def __repr__(self) -> str:
        return f"MLDataset(name={self.name!r}, size={len(self.data)})"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int):
        return self.data[index]
    
    def __contains__(self, item) -> bool:
        return item in self.data
    
# Test it

ds = MLDataset("Training", [10, 20, 30, 40, 50])

print(ds)       # uses __repr__
print(len(ds))  # uses __len__
print(ds[2])    # uses __getitem__
print(20 in ds) # uses __contains__

for item in ds: # works automatically because __len__ + __getitem__ exist
    print(item)

