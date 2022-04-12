import torch

def norm2(x):
    ord = 2
    return torch.pow(torch.sum(torch.pow(torch.abs(x), ord)), (1 / ord))

def low_rank_approximation(F):
    # F.shape = (B, T, H)
    batch_size = F.size(0)
    m = F.size(1)
    E_batched = torch.zeros_like(F)
    for b in range(batch_size):
        E = F[b,0,:] # (B, H)
        print(E.shape, E.t().shape)
        M = 1./torch.matmul(E, E.t()) # (1,1)
        E = E.unsqueeze(0)
        #M = 1./torch.einsum('bnh,bnh -> bb', [E, E])(E, E.t()) # (B,B)
        l = []
        print('F', F.shape)
        print('M', M.shape)
        print('E', E.shape)
        print('m', m)
        error0 = 0.001
        for k in range(2,m):
            print('----------')
            fc = F[:,k,:] # (B, H)
            print(k, 'fc', fc.shape)
            print('M', M.shape, 'E', E.shape, 'fc', fc.shape)

            #beta = torch.einsum('bb,tbh,bh -> b', [M, E, fc]) # (B,)
            beta = torch.matmul(torch.matmul(M, E.transpose(0,1)), fc) # (B,)
            print(k, 'beta', beta.shape)
            print('M', M.shape)
            print('E', E.shape)
            print('fc', fc.shape)
            error_ = torch.einsum('jbh,bb,jbh,bh -> b', [E, M, E, fc])
            """ error = norm2(
                fc-torch.matmul(torch.matmul(torch.matmul(E, M), E.t()), fc),
            ) """ # (B,)
            error = norm2(error_)
            if error > error0:
                print(k, 'E', E.shape, 'fc', fc.unsqueeze(0).shape)
                E = torch.cat((E, fc.unsqueeze(0)), dim=0)
                print(k, 'post E', E.shape)
                print(torch.add(M,torch.matmul(beta.t(), beta)).shape)
                M = torch.as_tensor([
                    [
                        torch.div(torch.add(M,torch.matmul(beta.t(), beta)), error),
                        -beta / error
                    ],
                    [
                        -beta.t() / error, 1/error
                    ]
                ])
                print(k, 'post M', M.shape)
                l.append(1)
            else:
                l.append(0)
        print(F.shape, E.shape)
        sys.exit()
        E_batched[b,...] = E
    return E_batched