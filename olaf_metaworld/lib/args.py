import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_traj", type=str, help="Specify input obs file", required=True)
    parser.add_argument("-o", "--output", type=str, help="Specify relabeled directory name", required=True)
    
    args = parser.parse_args()
    
    return args