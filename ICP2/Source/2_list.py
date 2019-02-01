A_list = ["1", "4", "0", "6", "9"]

# Another way to do this: Int_list = list(map(int, A_list))
Int_list = [int(x) for x in A_list]

Sort_list = sorted(Int_list)

print(A_list)
print(Sort_list)
