#!/bin/bash
#SBATCH -A research 
#SBATCH --qos=medium
#SBATCH -c 10
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -o ml4ns_mdcnn.txt 
#SBATCH --job-name=ML4NS_MDCNN

echo "Job Started"
cd  /home2/kushagra2443/MultiDigitCNN
echo "Running Script"
python3 train_data.py
echo "Job Finished"

