local regimes = {
          --nEpoch,    LR,   WD,
        { 5, 1e-5,  0.03 },
        { 3, 1e-3,  1e-3 },
        { 2, 3e-3,  1e-3 },
        { 20, 3e-3,  1e-4 },
        { 80, 3e-3,  0 },
        { 30,  1e-3,  0 },
        { 20,  1e-4,  0 },
        }
opt.nEpochs = 160
return regimes
