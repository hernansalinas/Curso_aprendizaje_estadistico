def doble_factorial(n):
    s=n
    while(n>2):
        n= n-2
        s=s*n
        print(n,s)

n=10
result=doble_factorial(n)
print("el doble factorial",n "es", result)
        
