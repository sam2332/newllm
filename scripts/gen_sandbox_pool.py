"""Precompute verified (question, command, output) triples for sandbox traces.

Executing Docker once per training trace would cost ~0.4s each, so instead we
run a fixed task list once, keep only the commands that actually succeed, and
sample from that pool when composing traces. Every stored output is real
container output - nothing is invented.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "/home/lmeadows/llm")
from agent.sandbox_tools import run_bash, run_python, docker_available

BASH_TASKS = [
    ("How many words are in the phrase 'the quick brown fox jumps'?",
     "echo 'the quick brown fox jumps' | wc -w"),
    ("Print the numbers 1 to 10 on one line.", "seq 1 10 | tr '\\n' ' '"),
    ("Sort these names alphabetically: delta, alpha, charlie, bravo.",
     "printf 'delta\\nalpha\\ncharlie\\nbravo\\n' | sort"),
    ("Count the characters in the word 'transformer'.",
     "echo -n 'transformer' | wc -c"),
    ("Show the first 3 even numbers between 1 and 20.",
     "seq 2 2 20 | head -3"),
    ("Reverse the string 'agent'.", "echo 'agent' | rev"),
    ("What is 2 to the power of 16, using the shell?",
     "python -c 'print(2**16)'"),
    ("Convert 255 to hexadecimal.", "printf '%x\\n' 255"),
    ("Count how many lines are in a 5-line file.",
     "seq 1 5 > /tmp/f && wc -l < /tmp/f"),
    ("Find the unique values in: a b a c b a.",
     "printf 'a\\nb\\na\\nc\\nb\\na\\n' | sort -u | tr '\\n' ' '"),
    ("What is today's date inside the sandbox?", "date +%Y-%m-%d"),
    ("Sum the numbers 1 through 100.",
     "seq 1 100 | paste -sd+ | bc"),
    ("Show the largest of these numbers: 42 17 93 8.",
     "printf '42\\n17\\n93\\n8\\n' | sort -n | tail -1"),
    ("Replace all spaces with underscores in 'hello big world'.",
     "echo 'hello big world' | tr ' ' '_'"),
    ("Count occurrences of the letter a in 'banana'.",
     "echo 'banana' | tr -cd 'a' | wc -c"),
]

PYTHON_TASKS = [
    ("What is 17 factorial?", "import math; print(math.factorial(17))"),
    ("List the first 10 Fibonacci numbers.",
     "a,b=0,1\nout=[]\nfor _ in range(10):\n    out.append(a); a,b=b,a+b\nprint(out)"),
    ("Is 97 a prime number?",
     "n=97\nprint(all(n%i for i in range(2,int(n**0.5)+1)))"),
    ("What is the square root of 2 to 8 decimal places?",
     "import math; print(f'{math.sqrt(2):.8f}')"),
    ("Sort this list by length: ['banana','fig','apple','kiwi'].",
     "print(sorted(['banana','fig','apple','kiwi'], key=len))"),
    ("What are the prime factors of 360?",
     "n=360; f=[]; d=2\nwhile d*d<=n:\n    while n%d==0: f.append(d); n//=d\n    d+=1\nif n>1: f.append(n)\nprint(f)"),
    ("Count the vowels in 'transformer architecture'.",
     "s='transformer architecture'; print(sum(c in 'aeiou' for c in s))"),
    ("What is the sum of squares from 1 to 20?",
     "print(sum(i*i for i in range(1,21)))"),
    ("Convert 1000 seconds to hours, minutes, seconds.",
     "t=1000; print(f'{t//3600}h {t%3600//60}m {t%60}s')"),
    ("Show the binary representation of 200.", "print(bin(200))"),
    ("What is the mean of [4, 8, 15, 16, 23, 42]?",
     "d=[4,8,15,16,23,42]; print(sum(d)/len(d))"),
    ("Reverse the words in 'one two three four'.",
     "print(' '.join('one two three four'.split()[::-1]))"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sandbox_pool.json")
    args = ap.parse_args()

    if not docker_available():
        print("docker unavailable; nothing to do")
        return

    pool = []
    for question, command in BASH_TASKS:
        out = run_bash(command)
        ok = not out.startswith("ERROR") and "exit code" not in out.split("\n")[0]
        print(f"[bash ] {'ok ' if ok else 'SKIP'} {question[:46]:46s} -> {out[:40]!r}")
        if ok and out.strip():
            pool.append({"tool": "run_bash", "question": question,
                         "args": {"command": command}, "output": out})
    for question, code in PYTHON_TASKS:
        out = run_python(code)
        ok = not out.startswith("ERROR") and "exit code" not in out.split("\n")[0]
        print(f"[py   ] {'ok ' if ok else 'SKIP'} {question[:46]:46s} -> {out[:40]!r}")
        if ok and out.strip():
            pool.append({"tool": "run_python", "question": question,
                         "args": {"code": code}, "output": out})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(pool, open(args.out, "w"), indent=1)
    print(f"\n{len(pool)}/{len(BASH_TASKS)+len(PYTHON_TASKS)} verified "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
