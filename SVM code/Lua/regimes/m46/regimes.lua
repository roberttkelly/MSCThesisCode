local regimes = {
          --nEpoch,    LR,   WD,
        { 5, 1e-4,  1e-3 },
        { 3, 1e-3,  1e-3 },
        { 2, 3e-3,  1e-3 },
        { 100, 3e-3,  0 },
        { 30,  1e-3,  0 },
        { 22,  1e-4,  0 },
        }
opt.nEpochs = 162
return regimes
