local regimes = {
          --nEpoch,    LR,   WD,
        { 5, 1e-4,  1e-3 },
        { 100,  0.01,  0 },
        { 30,  3e-3,  0 },
        { 30,  1e-3,  0 },
        { 20,  1e-4,  0 },
        }
opt.nEpochs = 185
return regimes
