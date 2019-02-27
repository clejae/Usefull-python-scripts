import os

os.chdir('B:/temp/temp_clemens/Validation/EnMAP-Pixel-Validation/')

# read estimate matrix
ds = open('combined_planeNumbers_B.csv')
ds_v1 = ds.readlines()

# convert each string into a list, remove linebreaks (\n) and replace commas with points
cleanList = []
for string in ds_v1:
    string = string.replace('\n','')
    string = string.replace(',','.')
    a = string.split(';')
    cleanList.append(a)

# transforms per class values into lists
#finalList = [['gra'],['shr'],['btr'],['con'],['soi'],['imp'],['other']]
finalList = [['gra'],['shr'],['btr'],['soi'],['imp'],['other']]
for subList in cleanList:
    for i in range(0,3):
        finalList[0].append(float(subList[i]))
    for i in range(3,6):
        finalList[1].append(float(subList[i]))
    for i in range(6,9):
        finalList[2].append(float(subList[i]))
    for i in range(9,12):
        finalList[3].append(float(subList[i]))
    for i in range(12,15):
        finalList[4].append(float(subList[i]))
    for i in range(15,18):
        finalList[5].append(float(subList[i]))
#    for i in range(18,21):
#        finalList[6].append(float(subList[i]))

# create list of unique IDs
roiIDList = ['UniqueID']
#for i in range(1,66):
#    for j in range(1,10):
#        uniqueID = 'A_'+str(i)+'_'+str(j)
#        roiIDList.append(uniqueID)
for i in range(1,51):
    for j in range(1,10):
        uniqueID = 'B_' + str(i) + '_' + str(j)
        roiIDList.append(uniqueID)

# append this unique ID list to the value list
finalList.append(roiIDList)

# for output change columns and rows
finalList2 = []
for i in range(len(finalList[0])):
    subList = []
    for j in range(len(finalList)):
        subList.append(finalList[j][i])
    finalList2.append(subList)

# write list to csv
outputFile = open("final_List_B.csv", 'w')
for item in finalList2:
    i=0
    for subitem in item:
        if i < len(finalList)-1:
            outputFile.write(str(subitem))
            outputFile.write(";")
            i += 1
        else:
            outputFile.write(str(subitem))
    outputFile.write('\n')
del outputFile

#dataTypeList = ["Real","Real","Real","Real","Real","Real","Real","String"]
dataTypeList = ["Real","Real","Real","Real","Real","Real","String"]
# write csvt file
outputFile  = open("final_List_B.csv", "w")
for item in dataTypeList:
    outputFile.write('"'+item+'"')
    outputFile.write(',')
del outputFile