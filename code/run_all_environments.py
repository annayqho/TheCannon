import os

os.system("conda create -n [name] python=2.7 etc matplotlib astropy")
os.system("source activate [name]")
os.system("python run_apogee_example.py")
os.system("source deactivate")

# it should work for: 1.7-0.13, 1.8-0.14, 1.9-0.15 
