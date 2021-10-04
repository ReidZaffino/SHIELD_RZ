#Comparison of Schoolbook and Karatsuba Multiplication Algorithm Speeds
#Reid Zaffino (2021-10-04)

#Import required libraries, multiplication algorithms, numpy for array manipulation, random for random numbers, and stopwatch for timing.
import KaratsubaMultiplication as km
import SchoolbookMultiplication as sm
import numpy as np
import random
from stopwatch import Stopwatch

def main():
    #Specified bit width of the multiplicand and multiplier.
    bitwidth = 93

    #Randomized numbers within the specified bitwidth, and specify number of trials.
    numOfTrials = 1000
    a = [random.randint(0, 2**bitwidth - 1) for _ in range(numOfTrials)]
    b = [random.randint(0, 2**bitwidth - 1) for _ in range(numOfTrials)]

    #Initialize the product array with a bitwidth of twice the specified bitwidth.
    product = np.zeros(bitwidth*2).astype(int)

    #Initialize the sum of times and the stopwatches.
    KaratsubaTime = 0
    SchoolbookTime = 0
    tkm = Stopwatch()
    tsm = Stopwatch()

    #For loop used to run algorithms for a given number of trials.
    for i in range(len(a)):
        #Create arrays for use in Schoolbook Multiplication algorithm.
        ArrA = sm.prepArray(a[i], bitwidth)
        ArrB = sm.prepArray(b[i], bitwidth)

        #Time and run Karatsuba Multiplication.
        tkm.start()
        kmproduct = km.karatsuba(a[i],b[i])
        tkm.stop()
        
        #Time and run Schoolbook Multiplication.
        tsm.start()
        smproduct = sm.schoolbook(ArrA, ArrB, product)
        tsm.stop()

        #Add time elapsed for trial to overall times.
        KaratsubaTime = KaratsubaTime + float(tkm.elapsed)
        SchoolbookTime = SchoolbookTime + float(tsm.elapsed)

        #Reset timers.
        tkm.reset()
        tsm.reset()

    #Output info about timing of algorithms.
    print("Average time for Karatsuba multiplication: " + str(KaratsubaTime/numOfTrials))
    print("Average time for Schoolbook multiplication: " + str(SchoolbookTime/numOfTrials))
    print("Karatsuba multiplication is performed in roughly " + str((KaratsubaTime/SchoolbookTime) * 100)[:5] + "% of the time it takes Schoolbook multiplication.")


if __name__ == "__main__":
    main()