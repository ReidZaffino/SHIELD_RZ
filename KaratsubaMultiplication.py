#Karatsuba Multiplication Algorithm
#Reid Zaffino (2021-10-03)

#Karatsuba multiplication algorithm (binary).
def karatsuba(x, y):
    #Base case condition in which multiplicand and multiplier are -1, 0, or 1, LUT for base case used.
    if (x > -2 and x < 2) and (y > -2 and y < 2):
        if((x == -1 and y == -1) or (x == 1 and y == 1)):
            return 1
        elif((x == -1 and y == 1) or (x == 1 and y == -1)):
            return -1
        else:
            return 0
    #Divide and conquer recursion of Karatsuba algorithm if not in base case.
    else:
        #Setting the sign of multplicand and multiplier as positive (True).
        xsign = True
        ysign = True
        #Adjusting the sign boolean values if neccessary and converting the numbers to binary strings.
        if (x < 0):
            xsign = False
            x = str(bin(x))[3:]
        else:
            x = str(bin(x))[2:]
        if (y < 0):
            ysign = False
            y = str(bin(y))[3:]
        else:
            y = str(bin(y))[2:]

        #Set the bitwidth of the multiplication.
        n = max(len(x), len(y))

        #Ensuring the bitwidth is of even length.
        if (n % 2 != 0):
            n = n + 1

        #Padding binary strings with zeros if neccessary to fit the same bitwidth.
        x = x.zfill(n)
        y = y.zfill(n)

        #Calculating half the bitwidth.
        m = n//2

        #Splitting multiplicand and multiplier into halves.
        a = x[:m]
        b = x[m:]
        c = y[:m]
        d = y[m:]

        #Calculating intermediate multiplication terms with recursive call to method.
        n0 = karatsuba(int(b, 2), int(d, 2))
        n1 = karatsuba(int(a, 2) - int(b, 2), int(d, 2) - int(c, 2))
        n2 = karatsuba(int(a, 2), int(c, 2))

        if (xsign == ysign):
            return ((n2 << n) + ((n1 + n2 + n0) << m) + n0)
        else:
            return -((n2 << n) + ((n1 + n2 + n0) << m) + n0)

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

    #Function call of Karatsuba algorithm.
    c = karatsuba(a, b)

    #Output of the product in integer and binary form.
    print("The product is " + str(c) + ", (" + str(bin(c)) + ")")

    #Assertion which will trigger an error if the multiplication is incorrect.
    assert c == a*b

if __name__ == "__main__":
    main()