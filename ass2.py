#creating sets
set1=set()
set2=set()

#adding set1
for i in range(1,6):
    set1.add(i)

#adding set2
for i in range(2,5):
    set2.add(i)

#displaying sets    
print("set1=",set1);
print("set2=",set2);

#union of sets
set3=set1 | set2
print("union of sets(set3)=",set3)

#intersection of sets
set4=set1 & set2
print("intersetion of sets(set4)=",set4)

#super set
if set3 > set4:
    print("Set 3 is superset of Set4")

elif set3 < set4:
    
    print("Set 3 is subset of Set4")
else :
    print("Set 3 is same as Set4")

#subset
if set4 < set3:
    print("set 4 is subset of set3")
    print("\n")

#difference
set5 = set3 - set4
print("Elements in Set3 are not in Set4 = ", set5)
print("\n")
