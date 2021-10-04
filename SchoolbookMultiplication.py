#Schoolbook Multiplication Algorithm
#Reid Zaffino (2021-10-01)

#Import required library numpy for arrays and number manipulation.
import numpy as np

#Schoolbook multiplication algorithm, also called long multiplication (binary).
def schoolbook(ArrA, ArrB, product):
    #Loop which iterates for the multiplier digits.
    for b_i in range(len(ArrB)):
    #Initial carry state set to zero.
        carry = 0
        #Loop which iterates for the multiplicand digits.
        for a_i in range(len(ArrA)):
            #Determine the product of specified digit.
            product[a_i + b_i] += carry + int(ArrA[a_i] and ArrB[b_i])
            #Adjust carry value.
            carry = int(product[a_i + b_i] // 2)
            #Ensure digit is binary.
            product[a_i + b_i] = product[a_i + b_i] % 2
        #Specify final digit is the ending carry.
        product[b_i + len(ArrA)] = carry
    
    #Reverse the array of the product so it can be read properly, and output as an integer.
    product = product[::-1]
    c = int("".join(str(i) for i in product), 2)
    return c

#Convert the multiplicand and multiplier to binary arrays, pad with additional zeroes and reverse the arrays.
def prepArray(a, bitwidth):
    ArrA = np.array([int(x) for x in str(bin(a))[2:]])
    if len(ArrA) < bitwidth:
        ArrA = np.pad(ArrA, (bitwidth - len(ArrA), 0), 'constant', constant_values=(0))
    ArrA = ArrA[::-1]
    return ArrA

def main():
    #Import required library random for random numbers.
    import random
    #Specified bit width of the multiplicand and multiplier.
    bitwidth = 93

    #Randomized numbers within the specified bitwidth.
    a = random.randint(0, 2**bitwidth - 1)
    b = random.randint(0, 2**bitwidth - 1)

    #Output of the multiplicand and multiplier in integer and binary form.
    print("The bitwidth is " + str(bitwidth))
    print("The first random number is " + str(a) + ", (" + str(bin(a)) + ")")
    print("The second random number is " + str(b) + ", (" + str(bin(b)) + ")")

    #Prepare the arrays of numbers.
    ArrA = prepArray(a, bitwidth)
    ArrB = prepArray(b, bitwidth)

    #Initialize the product array with a bitwidth of twice the specified bitwidth.
    product = np.zeros(bitwidth*2).astype(int)

    #Function call of Karatsuba algorithm.
    c = schoolbook(ArrA, ArrB, product)

    #Output of the product in integer and binary form.
    print("The product is " + str(c) + ", (" + str(bin(c)) + ")")

    #Assertion which will trigger an error if the multiplication is incorrect.
    assert c == a*b

if __name__ == "__main__":
    main()