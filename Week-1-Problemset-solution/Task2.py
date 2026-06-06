
# This is about matrix operations 

# A . convert transpose of matrix ! By swapping element i guess instead 

rows = int(input("Rows: "))

matrix = [list(map(int, input().split())) for _ in range(rows)]

for i in range(rows):
    for j in range(i + 1, rows):
        matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

print(matrix)

#B. diagonal sum

n = int(input("Number-Of-Rows: "))

arr = [list(map(int, input().split())) for _ in range(n)]

sum1 = 0
sum2 = 0

for i in range(n):
    for j in range(n):
        if i == j:
            sum1 += arr[i][j]
        elif i + j == n - 1:
            sum2 += arr[i][j]

print(sum1, sum2, sep="\n")
