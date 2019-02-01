# set new empty set for output
output = []
start = 100

while start < 500:
    odd = start
    tot = 0
    while odd > 0:
        dig = odd % 10
        if dig % 2 != 0:
            tot = tot + 1
        odd = odd // 10
    if tot == 3:
        output.append(start)
    start = start + 1

print(output)
