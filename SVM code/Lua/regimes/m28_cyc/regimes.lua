local regimes = {
          --nEpoch,    LR,   WD,
        { 2, 1e-3,  1e-3 },
        { 50,  0.01,  0 },
        { 30,  3e-3,  0 },
        { 20,  1e-3,  0 },
        { 10,  1e-4,  0 },
        }
opt.nEpochs = 112
return regimes
