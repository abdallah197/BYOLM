import sys


def solution(A):
  """Your solution goes here."""
  
    if not A or len(A) ==1:
        return len(A)
    dicplus = {}
    dicminus = {}
    for i , n in enumerate(A):
        if n[0] == "+":
            dicplus[n[1:]] = dicplus.get(dicplus[n[1:]] +1 ,0)
        elif n[0] == "-":
            dicminus[n[1:]] = dicminus.get(dicminus[n[1:]] +1 ,0)
    count = 0
    for item in dicplus:
        if item in dicminus:
            count += 



def main():
  # Read from stdin, solve the problem, and write the answer to stdout.
  A = [s[1:-1] for s in sys.stdin.readline()[1:-1].split(",")]
  sys.stdout.write(str(solution(A)))


if __name__ == "__main__":
  main()
