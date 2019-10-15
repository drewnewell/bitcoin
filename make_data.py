import click
import pickle
import struct

from pifkoin.blockchain import BlockHeader


OUTPUT_FILE = 'data/nonce.pkl'
BITCOIN_CONF_FILE = '/home/drew/archive/bitcoin/bitcoin.conf'


@click.command()
@click.option('--output', help='output file', default=OUTPUT_FILE)
@click.option('--height', help='starting height', type=int)
@click.option('--size', help='number of blocks to extract', type=int, default=1000)
@click.option('--conf', help='configuration file', default=BITCOIN_CONF_FILE)
def extract(output, height, size, conf):

    def get_blockheader(height):
        return BlockHeader.from_blockchain(height=height, config_filename=conf)

    last_block = get_blockheader(-1)
    if height is not None:
        current_height = height
    else:
        current_height = last_block.height
    print('starting with block {}'.format(current_height))

    data = []
    for i in range(size):
        print('processing block {}...\r'.format(current_height - i), end='')
        header = get_blockheader(current_height - i)
        header_bytes = header.bytes
        #parital_bytes = header_bytes[:-4]
        #nonce = header_bytes[-4:]

        data.append(header_bytes)

    with open('data/' + output, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    extract()
