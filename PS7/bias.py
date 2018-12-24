import random

def generateEs():
    e1 = random.uniform(0, 1)
    e2 = random.uniform(0, 1)
    e = min(e1, e2)
    return e1, e2, e

totale1 = 0
totale2 = 0
totale = 0
for i in range(100000):
    e1, e2, e = generateEs()
    totale1 += e1
    totale2 += e2
    totale += e
print(totale1 / 100000, totale2 / 100000, totale / 100000)
