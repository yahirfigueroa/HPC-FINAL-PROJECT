import sys
import random

#Yahir Figueroa Vega

if len(sys.argv) != 3:
    print("Usage: python generate_simple_realistic.py <n> <m>")
    sys.exit(1)

n= int(sys.argv[1])
m = int(sys.argv[2])
#assigns popularity range
min_popularity = 10
max_popularity = 100

item_popularity=[]

#generate random good popularity
for i in range(m):
    base=random.randint(min_popularity, max_popularity)
    item_popularity.append(base)

with open("matrix_300x20000.txt", "w") as file:
    file.write(f"{n} {m}\n")
    for i in range(n):
        row = []
        for j in range(m):
            #random noise to simulate distinct valuations
            noise = random.randint(0, 20)
            row.append(str(item_popularity[j] + noise))
        file.write(" ".join(row) + "\n")
print("Generated matrix correctly")