from typing import Generator

def stream_csv(filepath: str) -> Generator[list, None, None]:
    """Read a csv file one row at a time - never loads full file.

    Args:
        filepath: Path to the csv file.

    Yields:
    Each row as a list of strings.
    """

    with open(filepath) as f:
        header = f.readline().strip().split(',')
        for line in f:
            values = line.strip().split(',')
            yield dict(zip(header, values))

def batch_generator(
        data: list,
        batch_size: int
) -> Generator[list, None, None]:
    """Yield items from a list in fixed-size batches.
    
    Args:
        data: Any list of items.
        batch_size: How many items per batch.
        
    Yields:
        Batches as sublists.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

