import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help='path to dir containing text files')
parser.add_argument('--seqlen', type=int, default=2048, help='number of character bytes per line')

def read_args():
    args = parser.parse_args()
    return args

def process(srcdir, seqlen):
    """
    Reads files in srcdir and writes contents to a single file with a fixed number of bytes per line
    """
    if not os.path.exists(srcdir):
        raise IOError('The source directory does not exist: {}'.format(srcdir))
    outpath = './traindata.txt'
    if os.path.exists(outpath):
        os.remove(outpath)
    with open(outpath, 'ab') as outfile:
        srcdocs = os.listdir(srcdir)
        for docnum, srcdoc in enumerate(srcdocs):
            print('Processing file {} of {}'.format(1 + docnum, len(srcdocs)))
            srcdoc = '{}/{}'.format(srcdir, srcdoc)
            doc = open(srcdoc, 'r').read()
            doc = doc.replace('\n', '\\n').replace('\r', '\\r')
            doc = doc.encode('utf-8')
            doc = [doc[i:i+seqlen] for i in range(0, len(doc), seqlen)]
            doc[-1] = doc[-1].ljust(seqlen)
            doc = [sequence + '\n'.encode('utf-8') for sequence in doc]
            outfile.writelines(doc)

if __name__ == '__main__':
    args = read_args()
    process(args.src, args.seqlen)
