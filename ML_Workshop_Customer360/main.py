import os
import subprocess
import sys


scripts = [
    os.path.join("src", "eda.py"),
    os.path.join("src", "Churn Classification.py"),
    os.path.join("src", "Revenue Regression.py"),
    os.path.join("src", "Unsupervised Customer.py"),
    os.path.join("src", "Dimensionality Reduction (PCA).py"),
]

for script in scripts:
    print("\n" + "=" * 70)
    print("RUNNING:", script)
    print("=" * 70)
   
    subprocess.check_call([sys.executable, script])

print("\nAll solution scripts completed successfully.")