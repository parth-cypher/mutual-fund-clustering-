# Dataset

`comprehensive_mutual_funds_data.csv` is the file from the Kaggle dataset
**"Mutual Funds India - Detailed"** (uploaded by ravibarnawal):
https://www.kaggle.com/datasets/ravibarnawal/mutual-funds-india-detailed

This copy was taken from the public GitHub repository https://github.com/nileshiq/Mutual-Funds-India
(commit `88cd08b8c451844a2aaad3528ad8600218a5f491`), whose README credits the Kaggle dataset above.
SHA-256: `2672b241c0715e4e099d81771b09da2108f5c2415693c0df2982cfc9ec155b5e`

- 814 rows (one per mutual fund scheme), 20 columns
- Values that are not available are written as `-`
- The code never modifies this file

If the file is missing, `python train.py` downloads the same copy and checks the SHA-256, or you can download it
from Kaggle (free account) and put it in this folder.
