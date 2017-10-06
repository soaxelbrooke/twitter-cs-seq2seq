""" Prep work to get data into x.txt and y.txt for training """

import csv

from tqdm import tqdm


def prep():
    """ Read CSV and write x.txt and y.txt """
    with open('data/twitter_cs.csv') as infile:
        reader = csv.reader(infile)
        headers = next(reader)
        rows = list(reader)

        with open('data/x.txt', 'w') as x_out:
            with open('data/y.txt', 'w') as y_out:
                for row in tqdm(rows):
                    x_out.write(f'@{row[1]} {row[2]}\n')
                    y_out.write(f'@{row[3]} {row[4]}\n')


if __name__ == '__main__':
    prep()
