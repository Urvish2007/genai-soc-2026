arr = input("Enter elements separated by spaces: ").split()

ans = []

for item in arr:
    if item not in ans:
        ans.append(item)

print("Duplicate-removed elements:", ans)