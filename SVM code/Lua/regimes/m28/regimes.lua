local regimes = {
          --nEpoch,    LR,   WD,
        { 2, 1e-3,  1e-3 },
        { 100,  0.01,  0 },
        { 50,  3e-3,  0 },
        { 18,  1e-3,  0 },
        { 20,  1e-4,  0 },
        }
opt.nEpochs = 190
return regimes
