import os.path
import sys
import time
from datetime import datetime
from os.path import dirname, join, splitext, getmtime
import click
from google.cloud import storage


class Monitor:
    def __init__(self, bucket, path, key_prefix):
        self.key_prefix = key_prefix
        self.bucket = bucket
        self.path = path
        self.prev_last = None

    def update(self):
        if not os.path.exists(self.path):
            return

        files = [ join(self.path, f) for f in os.listdir(self.path) ]
        last = max(files, key=lambda f: getmtime(f))

        if last == self.prev_last:
            return
        self.prev_last = last

        _, ext = splitext(last)
        date_str = datetime.fromtimestamp(getmtime(last)).strftime("%Y_%m_%d_%H_%M_%S_%f")
        key = f'{self.key_prefix}_{date_str}{ext}'

        blob = self.bucket.blob(key)
        if not blob.exists():
            print(f"Uploading file {last} to: {self.bucket.path}/{key}")
            blob.upload_from_filename(last)

        for old_file in files[0:-4]:
            print("Removing old file:", old_file)
            os.remove(old_file)


@click.command()
def main():
    base_path = dirname(__file__)
    ckpt_dir = join(base_path, 'checkpoint')
    samples_dir = join(base_path, 'sample')

    date_str = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')

    bucket = storage.Client().get_bucket('tokimeki-waifu')
    monitors = [
        Monitor(bucket, ckpt_dir, f'sg2/ckpt/{date_str}/ckpt'),
        Monitor(bucket, samples_dir, f'sg2/samples/{date_str}/samples')
    ]

    while True:
        for monitor in monitors:
            monitor.update()
        sys.stdout.write('.')
        sys.stdout.flush()
        time.sleep(60)


if __name__ == '__main__':
    main()