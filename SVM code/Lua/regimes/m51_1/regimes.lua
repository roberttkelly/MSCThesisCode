local regimes = {
          --nEpoch,    LR,   WD,
        --{ 5, 3e-3,  0 },
        { 30,  1e-3,  0 },
        { 30,  3e-4,  0 },
        { 30,  1e-4,  0 },
        { 30,  3e-5,  0 },
        { 1000,  1e-5,  0 }
    }
opt.nEpochs = 550
return regimes
