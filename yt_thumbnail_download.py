from asyncio import sleep
import requests
import os
import csv


def download_thumbnail(video_id, filename: str, dir: str = 'data/thumbnails/'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    url = f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg'
    if os.path.exists(os.path.join(dir, filename)):
        print(f'File {filename} already exists')
        return
    r = requests.get(url)
    if r.status_code == 200:
        with open(os.path.join(dir, filename), 'wb') as f:
            f.write(r.content)
    else:
        print(f'Error: {r.status_code}')


def download_csv(filename: str, id_column: int, output_dir: str = 'data/thumbnails/'):
    with open(filename, 'r', encoding="utf8") as f:
        reader = csv.reader(f)
        next(reader)
        test_data = [row for row in reader]
        data_len = len(test_data)
        for i, row in enumerate(test_data):
            video_id = row[id_column]
            filename = f'{video_id}.jpg'
            download_thumbnail(video_id, filename, output_dir)
            print(f'{i + 1}/{data_len} done')


def main():
    # with open('data/videos-stats.csv', 'r', encoding="utf8") as f:
    #     reader = csv.reader(f)
    #     next(reader)
    #     test_data = [row for row in reader]
    #     data_len = len(test_data)
    #     for i, row in enumerate(test_data):
    #         video_id = row[2]
    #         filename = f'{video_id}.jpg'
    #         download_thumbnail(video_id, filename)
    #         print(f'{i + 1}/{data_len} done')
    #         sleep(0.1)
    download_csv('data/CAvideos.csv', 0, "data/test")

    pass


if __name__ == '__main__':
    main()
