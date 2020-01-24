#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import shelve
from terminaltables import AsciiTable

parser = argparse.ArgumentParser(\
        prog='Prints Pickle Readme File',\
        description=''
        )


parser.add_argument('-loc', default='./train_data/', help='Path to directory containing readme')
parser.add_argument('-name', default='readme', help='name of readme')



args = parser.parse_args()
name = str(vars(args)['name'])
path = vars(args)['loc'] + '/' + name

data = []

print('----------------------------------------------------------------')
print('Fetching training info from: ', path)
print('----------------------------------------------------------------')
with shelve.open(path) as db:
    # print("{:<30} {:<10}".format('Label','Value'))
    for key,value in db.items():
        # print("{:30} {:<10}".format(key, value))
        data.append([str(key),str(value)])


table  = AsciiTable(data)
table.inner_row_border = True
print(table.table)







db.close()
