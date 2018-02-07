# import urllib
import numpy
# from tempfile import mkstemp
# from shutil import move
# from os import fdopen, remove
# import pandas as pd
import csv
def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

def remove_col(file_path):
    f=pd.read_csv(file_path)
    keep_col = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10']
    new_f = f[keep_col]
    new_f.to_csv("new-breast-cancer-wisconsin.data", index=False)

def reorder(file_path):
    with open(file_path, 'r') as infile, open('reordered.csv', 'a') as outfile:
        # output dict needs a list for new column ordering
        fieldnames = ['c10','c1','c2','c3','c4','c5','c6','c7','c8','c9']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        # reorder the header first
        writer.writeheader()
        for row in csv.DictReader(infile):
            # writes the reordered rows to the new file
            writer.writerow(row)

def change_rows(file_path):
    r = csv.reader(open(file_path))
    lines = list(r)
    for row in lines:
        print "Org: " + str(row)
        if row[0] == 'Iris-setosa':
            row[0] = 1
        else:
            row[0] = 0

        if float(row[1]) > 5.8:
            row[1] = 1
        else:
            row[1] = 0

        if float(row[2]) > 3.0:
            row[2] = 1
        else:
            row[2] = 0

        if float(row[3]) > 3.7:
            row[3] = 1
        else:
            row[3] = 0

        if float(row[4]) > 1.2:
            row[4] = 1
        else:
            row[4] = 0

        # print "PRO: " + str(row)

    writer = csv.writer(open('new-iris.csv', 'w'))
    writer.writerows(lines)
    
filepath = 'iris.csv'
change_rows(filepath)
# replace(filepath,'republica0','0')
