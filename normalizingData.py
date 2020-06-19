
import csv
fil=open('dataNormalized.csv', 'a', newline='') 
writer = csv.writer(fil)

with open('driving_log.csv', 'r', newline='')  as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        normVal = float(row[3])
        writer.writerow([row[0], round(normVal,7)])


with open('driving_log2.csv', 'r', newline='')  as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        normVal = float(row[3])
        writer.writerow([row[0], round(normVal,7)])


with open('driving_log3.csv', 'r', newline='')  as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        normVal = float(row[3])
        writer.writerow([row[0], round(normVal,7)])
