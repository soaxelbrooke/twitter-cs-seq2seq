""" Prep work to get data into x.txt and y.txt for training """

import csv
import sys

import psycopg2
from tqdm import tqdm


def prep(mode: str):
    """ Read CSV and write x.txt and y.txt """
    if mode == 'fromdb':
        write_csv_from_db('data/twitter_cs.csv')

    with open('data/twitter_cs.csv') as infile:
        reader = csv.reader(infile)
        headers = next(reader)
        rows = list(reader)

        with open('data/x.txt', 'w') as x_out:
            with open('data/y.txt', 'w') as y_out:
                for row in tqdm(rows):
                    x_text = row[2].strip().replace("\n", " ")
                    y_text = row[4].strip().replace("\n", " ")
                    x_out.write(f'{x_text}\n')
                    y_out.write(f'{y_text}\n')


def write_csv_from_db(out_path: str):
    """ Pulls data from database and writes CSV with columns:
        date,request_screen_name,request_text,reply_screen_name,reply_text
    """
    query = """
        SELECT
           request.created_at AS date,
           request.data #>> '{user,screen_name}' AS request_screen_name,
           request.data ->> 'text' AS request_text, 
           reply.data #>> '{user,screen_name}' AS reply_screen_name,
           reply.data ->> 'text' AS reply_text
        FROM tweets reply 
          INNER JOIN tweets request 
          ON reply.data ->> 'in_reply_to_status_id' = request.status_id
        WHERE request.data ->> 'in_reply_to_status_id' IS NULL;
    """
    conn = psycopg2.connect(dbname='twitter_cs')
    crs = conn.cursor()

    header = 'date,request_screen_name,request_text,reply_screen_name,reply_text'.split(',')

    with open(out_path, 'w') as outfile:
        writer = csv.writer(outfile)
        crs.execute(query)
        writer.writerow(header)
        for _row in tqdm(crs):
            row = [*_row]
            row[2] = row[2].replace('\n', ' ')
            row[4] = row[4].replace('\n', ' ')
            writer.writerow(row)


if __name__ == '__main__':
    prep(sys.argv[-1])
