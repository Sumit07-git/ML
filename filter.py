def even(num):
    if num%2==0:
        return True

lst=[2,4,5,6,8,24,56]
df=list(filter(even,lst))
print(df)

a=list(filter(lambda num:num%2==0,lst))
print(a)
