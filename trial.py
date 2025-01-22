s = 'atharvkumar9569394675'

s1 = sorted(s)

d = {}
for x in s1:
    if x not in d:
        d[x] = 1
    else:
        d[x] += 1


d1 = []
d2 = []

for x in s1:
    if x.isdigit():
        d1.append(x)
    else:
        d2.append(x)

d3 = d2 + d1

s2 =''
for x in d3:
    s2 = s2 + x



s3 = s2[0]
print(s3 + '=' + str(d[s3]))

for x in d3:
    if x == s3:
        continue
    else:
        print(x + '=' + str(d[x]))
        s3 = x
        


    









