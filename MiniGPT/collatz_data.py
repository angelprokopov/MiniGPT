import random
import sys

def collatz_step(n: int) -> int:
    return n // 2 if n % 2 == 0 else 3 * n + 1

def collatz_sequence(n: int):
    seq = [n]
    while n != 1:
        n = collatz_step(n)
        seq.append(n)
    return seq

def main(max_n=20000, path="collatz.txt",seed=0):
    random.seed(seed)
    lines = [f"{n}>{collatz_step(n)}" for n in range(1, max_n + 1)]
    random.shuffle(lines)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} lines to {path}")
    print(f"Example full sequence for 27: {len(collatz_sequence(27)) - 1}, steps to reach 1")

if __name__ == "__main__":
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    main(max_n)