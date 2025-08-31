def even_odd(num):
    if num%2==0:
        return "The number {} is Even".format(num)
    else:
        return "The number {} is Odd".format(num)

lst=[2,45,34,62,22,78,66]
df=list(map(even_odd,lst))
print(df)