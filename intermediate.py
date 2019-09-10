import hashlib
import struct

from pifkoin.blockchain import BlockHeader
from pifkoin.sha256 import SHA256


BITCOIN_CONF_FILE = '/home/drew/archive/bitcoin/bitcoin.conf'

def get_blockheader(height):
    return BlockHeader.from_blockchain(height=height, config_filename=BITCOIN_CONF_FILE)

last_block = get_blockheader(-1)
current_height = last_block.height

print('hashing block ', current_height)

round_offset = 0


class traceHash(SHA256):

    @classmethod
    def _process_block(cls, message, state, round_offset=0):

        print(f"processing block")

        w = cls._expand_message(struct.unpack('>LLLLLLLLLLLLLLLL', message))

        midstate = state
        for i in range(64):
            midstate = cls._round(round_offset + i, w[i], midstate)
        return cls._finalize(midstate, state)

    def get_blocks(self):
        return [block for block in self._pad_message(self.buffer, self.length)]


first_hash = traceHash(last_block.bytes).digest()
bitcoin_state = traceHash(first_hash).hexdigest()

bitcoin_check = hashlib.sha256(hashlib.sha256(last_block.bytes).digest()).hexdigest().encode()

assert bitcoin_state == bitcoin_check
