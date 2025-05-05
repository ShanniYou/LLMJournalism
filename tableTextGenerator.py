# Shanni You, 04/10/2025
# This is the main function to convert relation table into text

# To do list:
# 1. Read through sql
# 2. Show databases
# 3. Loop though the table
# 4. Try chunk size, table: 101316_car_accidents
# 5. automated

from sqlUtils import *

def tableSparser(cTable):
    pass

def main():
    cTable, colName = table() # Currently fetching all the information
    print(type(cTable[-1]))
    print(cTable[-1][0], colName)

    tableName = 'car accidents'
    paramID = colName[0]
    

if __name__ == '__main__':
    main()




