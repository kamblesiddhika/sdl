def word_Count(str):
    count=dict()
    w=str.split()
    for a in w:
        if a in count:
            count[a]=count[a]+1
        else :
            count[a]=1

    return count

str1=input("Enter a string::")
c=word_Count(str1)

print(c)
#print(sorted(c))
d=sorted(c.items())
print(d)
