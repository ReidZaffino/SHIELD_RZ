#Schoolbook Multiplication Algorithm
#Reid Zaffino (2021-10-01)

#Import required libraries, numpy for arrays and number manipulation, and random for random numbers.
import numpy as np
import random

#Specified bit width of the multiplicand and multiplier.
bitwidth = 93

#Randomized numbers within the specified bitwidth.
a = random.randint(0, 2**bitwidth)
b = random.randint(0, 2**bitwidth)

#Output of the multiplicand and multiplier in integer and binary form.
print("The bitwidth is " + str(bitwidth))
print("The first random number is " + str(a) + ", (" + str(bin(a)) + ")")
print("The second random number is " + str(b) + ", (" + str(bin(b)) + ")")

#Convert the multiplicand and multiplier to binary arrays, pad with additional zeroes and reverse the arrays.
ArrA = np.array([int(x) for x in str(bin(a))[2:]])
if len(ArrA) < bitwidth:
    ArrA = np.pad(ArrA, (bitwidth - len(ArrA), 0), 'constant', constant_values=(0))
ArrA = ArrA[::-1]

ArrB = np.array([int(x) for x in str(bin(b))[2:]])
if len(ArrB) < bitwidth:
    ArrB = np.pad(ArrB, (bitwidth - len(ArrB), 0), 'constant', constant_values=(0))
ArrB = ArrB[::-1]

#Initialize the product array with a bitwidth of twice the specified bitwidth.
product = np.zeros(bitwidth*2).astype(int)

#Schoolbook multiplication algorithm, also called long multiplication.
#Loop which iterates for the multiplier digits.
for b_i in range(len(ArrB)):
    #Initial carry state set to zero.
    carry = 0
    #Loop which iterates for the multiplicand digits.
    for a_i in range(len(ArrA)):
        #Determine the product of specified digit.
        product[a_i + b_i] += carry + ArrA[a_i] * ArrB[b_i]
        #Adjust carry value.
        carry = int(product[a_i + b_i] / 2)
        #Ensure digit is binary.
        product[a_i + b_i] = product[a_i + b_i] % 2
    #Specify final digit is the ending carry.
    product[b_i + len(ArrA)] = carry

#Reverse the array of the product so it can be read properly, and store in separate variable as an integer.
product = product[::-1]
c = int("".join(str(i) for i in product), 2)

#Assertion which will trigger an error if the multiplication is incorrect.
assert c == a*b

#Output of the product in integer and binary form.
print("The product is " + str(c) + ", (" + str(bin(c)) + ")")