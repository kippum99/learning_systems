import numpy as np

def importData(filename):
    data = open(filename, 'r')
    lines = data.read().splitlines()

    matx = np.zeros(shape=(len(lines),3))
    vecty = np.zeros(shape=(len(lines),1))


    for i in range(len(lines)):
        point = lines[i].strip(' ').split('  ')
        matx[i] = np.matrix([1, float(point[0]), float(point[1])])
        vecty[i] = float(point[2])

    return matx, vecty


def transform(dataset):
    matx, vecty = dataset
    matz = np.zeros(shape=(len(matx), 8))
    for i in range(len(matx)):
        x1 = matx[i, 1]
        x2 = matx[i, 2]

        matz[i] = np.matrix([1, x1, x2, x1 ** 2, x2 ** 2, x1 * x2,
            abs(x1 - x2), abs(x1 + x2)])

    return matz

def calculateError(w, dataset):
    matz, vecty = dataset
    errorCount = 0
    for i in range(len(matz)):
        if np.sign(w.transpose().dot(matz[i, :])) != np.sign(vecty[i]):
            errorCount += 1

    return errorCount / len(matz)

#2
def transformLinReg():
    dataset = importData('in.dta.txt')
    matx, vecty = dataset
    matz = transform(dataset)

    #find w
    w = np.linalg.pinv(matz).dot(vecty)

    #calculate E_in
    inError = calculateError(w, (matz, vecty))

    #calculate E_out
    testData = importData('out.dta.txt')
    testMatz = transform(testData)
    outError = calculateError(w, (testMatz, testData[1]))

    print(inError, outError)

transformLinReg()


#3-6
def weightDecay(k):
    dataset = importData('in.dta.txt')
    matx, vecty = dataset
    matz = transform(dataset)

    #find w_reg
    lam = 10 ** k
    w = (np.linalg.inv(matz.transpose().dot(matz) + lam * np.identity(8))
        .dot(matz.transpose()).dot(vecty))

    #calculate E_in
    inError = calculateError(w, (matz, vecty))

    #calculate E_out
    testData = importData('out.dta.txt')
    testMatz = transform(testData)
    outError = calculateError(w, (testMatz, testData[1]))

    print(inError, outError)

#3
weightDecay(-3)

#4
weightDecay(3)

#5
for i in range(-2, 3):
    weightDecay(i)

#6
for i in range(-10, 10):
    weightDecay(i)
