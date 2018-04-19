import glob
import os

pattern = 'RESULTS/create_truth.*/start_dakota_summit_grid.sh'

files = glob.glob(pattern)

lines = []

with open('launch_mini_grids.sh', 'w') as f:
    for file in files:
        folder = os.path.split(file)[0]
        f.write('cd ' + folder + '; sbatch start_dakota_summit_grid.sh\n')
