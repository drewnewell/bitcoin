from __future__ import print_function

from pifkoin.blockchain import BlockHeader
import pickle
import struct


OUTPUT_FILE = 'data/nonce.pkl'
START_HEIGHT = 400000
BITCOIN_CONF_FILE = '/home/drew/archive/bitcoin/bitcoin.conf'


def main():

    def get_blockheader(height):
        return BlockHeader.from_blockchain(height=height, config_filename=BITCOIN_CONF_FILE)

    last_block = get_blockheader(-1)
    current_height = last_block.height

    output = []
    for i in range(current_height - START_HEIGHT):
        print('processing block {}...\r'.format(START_HEIGHT + i), end='')
        header = get_blockheader(START_HEIGHT + i)
        header_bytes = header.bytes
        parital_bytes = header_bytes[:-4]
        nonce = header_bytes[-4:]

        output.append((header.height, parital_bytes, nonce))

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(output, f)


if __name__ == '__main__':
    main()

