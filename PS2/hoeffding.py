import random

def flipCoins():
    randIndex = random.randint(0, 999)
    nuMin = 2
    for i in range(0, 1000):
        headCount = 0
        for j in range(0, 10):
            headCount += random.randint(0, 1)
        headFreq = headCount / 10
        if i == 1:
            nu1 = headFreq
        if i == randIndex:
            nuRand = headFreq
        if headFreq < nuMin:
            nuMin = headFreq
    return nu1, nuRand, nuMin

sumNu1 = 0
sumNuRand = 0
sumNuMin = 0
for i in range(0, 100000):
    nu1, nuRand, nuMin = flipCoins()
    sumNu1 += nu1
    sumNuRand += nuRand
    sumNuMin += nuMin
print((sumNu1/100000, sumNuRand/100000, sumNuMin/100000))
