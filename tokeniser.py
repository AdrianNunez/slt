import sentencepiece as spm
from transformers import MBartTokenizer

class Tokeniser:
    def __init__(self, cfg):
        super(Tokeniser).__init__()
               
        self.UNK_TOKEN = '<unk>'
        self.BOS_TOKEN = '<s>'
        self.EOS_TOKEN = '</s>'
        self.PAD_TOKEN = '<pad>'
        self.token_to_id = {}
        self.id_to_token = {}
        
        self.use_pretrained_embeddings = cfg["use_pretrained_embeddings"]
        if self.use_pretrained_embeddings: # not tested
            pretrained_embeddings_vocab = cfg['pretrained_embeddings_vocab']
            self.mapping = {}
            with open(pretrained_embeddings_vocab, 'r') as f:
                for count, line in enumerate(f):
                    id, token = line.strip().split('\t')
                    self.mapping[id] = count
             
            self.tokeniser = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
        else:
            vocab_path = cfg['txt_vocab']
            with open(vocab_path, 'r') as f:
                for line in f:
                    id, token = line.strip().split(' ')
                    new_idx = len(self.id_to_token)
                    self.id_to_token[new_idx] = token
                    self.token_to_id[token] = new_idx#int(id)
            tokeniser_file = cfg['tokeniser_path']
            self.tokeniser = spm.SentencePieceProcessor()
            self.tokeniser = spm.SentencePieceProcessor()
            self.tokeniser.Load(tokeniser_file)

    def encode(self, x):
        y = self.tokeniser.encode_as_ids(x)
        if self.use_pretrained_embeddings:
            y = [self.mapping[elem] for elem in y]
        return y

    def decode(self, x):
        if isinstance(x[0], int):
            decoded_txt = []
            for id in x:
                token = self.tokeniser.id_to_piece(int(id))
                if token == '</s>': break
                decoded_txt.append(token)
        else:
            decoded_txt = []
            for i in range(len(x)):
                sent = []
                for id in x[i]:
                    token = self.tokeniser.id_to_piece(int(id))
                    if token == '</s>': break
                    sent.append(token)
                decoded_txt.append(sent)
            
        return decoded_txt

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