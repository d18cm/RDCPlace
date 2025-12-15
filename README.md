# RDCPlace

## 📁 Project Structure

Plaintext

```
RDCPlace/
├── PPO_train_fast_stand.py    # [Entry Point] Main training script (arg parsing, training loop, logging)
├── benchmark/
│   └── ispd05/                # [Data] Download ISPD 2005 benchmarks (adaptec1, etc.) here
│   	├── adaptec1
│		└──	  ...
├── model/
│   └── PPO_agent_fast_stand.py # [Core] PPO Agent implementation (Actor and RDC2/Critic networks)
├── place_env/
│   ├── fast_env.py            # [Environment] GPU-accelerated Gym environment based on PyTorch
│   └── __init__.py            # Environment registration and utility files
└── utils/
    ├── place_db_bookshelf2.py # Data Loading: Parses ISPD05/Bookshelf formats
    ├── comp_res.py            # Evaluation metrics
    └── save_placement_pl.py   # Result Saving: Exports .pl placement files
```

## 🛠️Requirements

```
pip install -r requirements.txt
```



## 🏃‍♂️How to Run

1. First, ensure you have built the environment according to the requirements.
2. Download the **ISPD2005 benchmark** suite.
3. Place the downloaded files into `benchmark/ispd05/`.

**Download Link:** http://www.cerc.utexas.edu/~zixuan/ispd2005dp.tar.xz



### Basic Run Command：

```
cd RDCPlace
python PPO_train_fast_stand.py --benchmark adaptec1 --seed 42
```

- Common Arguments

  - `--benchmark`: Name of the test case (Default: `adaptec1`).
  - `--pnm`: Number of macros to place (Default: `2000`).
    - *Note: If this value exceeds the number of macros in the dataset, the actual count from the dataset will be used.*
  - `--grid`: Grid size (Default: `224`).
  - `--seed`: Random seed (Default: `42`).
  - `--train_epoch_max`: (Default `1000`).
  - `--load_model_path`: Path to load a pre-trained model (Optional).

  > **Note:** For a full list of arguments, please refer to the `parse_args()` function inside `PPO_train_fast_stand.py`.

### Output Results：

Training logs, model checkpoints (`.pth`), placement results (`.pl`), and visualization images will be saved in the `results/` directory.

The folder naming convention is: `{benchmark}_{pnm}_{seed}_{time}_{SMInfo}`



### Our Data

We provide a complete training record for `adaptec1` in the `experiment_logs/` directory. Additionally, the final `.pl` files for **all datasets** are provided in the `results/placement_pl/` folder.