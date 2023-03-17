"""
reddit_data_extractor.py
------------------------
This function extracts images from a pushshift reddit data file. The 
files can be downloaded from 'files.pushshift.io/reddit/comments/'.
"""
import argparse
import io
import json
import requests

from os import listdir
from os.path import isfile, join

import zstandard
import shutil

from imgur_downloader import ImgurDownloader
import re

def get_files(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

def extract_classes(jsonpath):
    location = dict()

    with open(jsonpath) as j:
        labels = json.load(j)

    for label, subreddits in labels.items():
        for s in subreddits:
            location[s] = label

    return location

def download_image(sample, location):
    url = None
    if re.match(r'https://i.redd.it/[a-zA-Z0-9]+.j?pn?g',sample['url']):
        url = sample['url']
        r = requests.get(url, stream = True)
        r.raw.decode_content = True
        with open('./Data/{loc}/{url}'.format(loc = location[sample['subreddit'].lower()], url = url.split("/")[-1]),'wb') as u:
            shutil.copyfileobj(r.raw, u)
    if sample['media']:
        if 'type' in sample['media']:
            if sample['media']['type'] == 'imgur.com': 
                #print(sample['media']['oembed'])
                if 'url' in sample['media']['oembed']:
                    r = sample['media']['oembed']['url']
                    url = re.findall(r'https://i\.imgur\.com/\S+\.j?pn?g', r)
                elif 'thumbnail_url' in sample['media']['oembed']:
                    r = sample['media']['oembed']['thumbnail_url']
                    url = re.findall(r'https://i\.imgur\.com/\S+\.j?pn?g', r)
                if url:
                    #print(url[0])
                    ImgurDownloader(url[0], 
                                    './Data/model_data/train/{}/'.format(location[sample['subreddit'].lower()])
                                ).save_images()

def main(args):
    files = get_files(args.directory)
    files.sort()

    location = extract_classes(args.classes)

    count = 0
    for f in files[1:3]:
        if args.verbose:
            print("Reading {}".format(f))
        with open('./Data/Compressed/'+f, 'rb') as compressed:
            decomp = zstandard.ZstdDecompressor(max_window_size=2147483648)
            stream_reader = decomp.stream_reader(compressed)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
            for line in text_stream:
                sample = json.loads(line)
                if sample['subreddit'].lower() in location:
                    download_image(sample, location)

                    count += 1
                    if args.verbose:
                        if count % 4000 == 0:
                            print(count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", action="store", 
                        help='Path to where data is stored'
                        )
    parser.add_argument("classes", action="store", 
                        help='Path to where subreddit json is stored'
                        )
    parser.add_argument("-v", "--verbose", action='store_true',
                        help='Execute verbosely'
                        )
    main(parser.parse_args())