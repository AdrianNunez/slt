import sentencepiece as spm

class Tokeniser:
    def __init__(self, cfg):
        super(Tokeniser).__init__()
        vocab_path = cfg['txt_vocab']
        
        self.UNK_TOKEN = '<unk>'
        self.BOS_TOKEN = '<s>'
        self.EOS_TOKEN = '</s>'
        self.PAD_TOKEN = '<pad>'
        """ self.token_to_id = {
            self.UNK_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.PAD_TOKEN: 3
        }
        self.id_to_token = {
            0: self.UNK_TOKEN,
            1: self.BOS_TOKEN,
            2: self.EOS_TOKEN,
            3: self.PAD_TOKEN
        } """
        self.token_to_id = {}
        self.id_to_token = {}
        with open(vocab_path, 'r') as f:
            for line in f:
                id, token = line.strip().split('\t')
                new_idx = len(self.id_to_token)
                self.id_to_token[new_idx] = token
                self.token_to_id[token] = new_idx#int(id)

        tokeniser_file = cfg['tokeniser_file']
        self.tokeniser = spm.SentencePieceProcessor()
        self.tokeniser = spm.SentencePieceProcessor()
        self.tokeniser.Load(tokeniser_file)

    def encode(self, x):
        pieces = self.tokeniser.encode_as_pieces(x)
        """ print(x, pieces)
        print([self.token_to_id.get(piece, self.unk_id()) for piece in pieces])
        sys.exit() """
        return [self.token_to_id.get(piece, self.unk_id()) for piece in pieces]

    def decode(self, x):
        return [self.id_to_token[id] for id in x]

    def __len__(self):
        return len(self.id_to_token)

    def bos_id(self):
        return self.token_to_id[self.BOS_TOKEN]

    def eos_id(self):
        return self.token_to_id[self.EOS_TOKEN]

    def pad_id(self):
        return self.token_to_id[self.PAD_TOKEN]

    def unk_id(self):
        return self.token_to_id[self.UNK_TOKEN]