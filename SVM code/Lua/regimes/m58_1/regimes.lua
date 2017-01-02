local regimes = {
          --nEpoch,    LR,   WD,
        { 10,  1e-3,  0 },
        { 15,  3e-4,  0 },
        { 15,  1e-4,  0 },
        { 10,  3e-5,  0 },
        { 1000,  1e-5,  0 }
    }
opt.nEpochs = 550
return regimes
