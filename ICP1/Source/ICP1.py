# Question1 State differences between Python 2 and Python 3 version
#
# Since Python 3 is the future, many of today's developers are creating libraries strictly for use with Python 3.
# Similarly, many older libraries built for Python 2 are not forwards-compatible.
#
# In Python 3, text strings are Unicode by default. In Python 2, strings are stored as ASCII by default
#
# The print statement has been replaced with a print() function in Python3

# Question2-1
inpt = input("Please enter your input: ")
print(inpt[::-1])

# Question2-2
print('Provide two integer A and B, get sum of them')
A1 = int(input('enter the first integer: '))
B1 = int(input('enter the second integer: '))
sum1 = A1 + B1
print("The sum of ", A1, "and ", B1, "is", sum1)

# Question3
user_inpt = input("Please enter your input: ")

numbers = sum(c.isdigit() for c in user_inpt)
letters = sum(c.isalpha() for c in user_inpt)
words = sum(c.isspace() for c in user_inpt) + 1

print("The number of letters is:", letters)
print("The number of words is:", words)
print("The number of digits is:", numbers)
