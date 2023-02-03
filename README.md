# Towards Fair Allocation In Social Commerce
## Use bellow commands to run algorithms
```bash
cd code_www
python [algoname].py [Instance] [L] [alpha] [Epsilon] [R2_option]
```
Where 
- Instance is the dataset instance.
- For alpha, take a value greator than 0 but less than or equal to 1. (0 <alpha<=1)
- R2_option = 1 for R2 = number-of-re-sellers, R2_option = 2 for R2 = 2 * R1, and R2_option = 3 for R2 = R1 + 1


for example:
To run SEAL use:
```bash
python SEAL.py 1 15 1 0 1
```

To run GreedyNash use:
```bash
python GreedyNash 1 15 1 0 1
```

To run NashMax use:
```bash
python NashMax.py.py 1 15 1 3 2
```

To run RevMax use:
```bash
python RevMax.py 1 15 1 3 2
```

Requirements
- Python3.6.9
- Gurobi Optimizer version 9.0.2 is required to run NashMax and RevMax.

Results are saved in results/[algo_name]/ folder
