import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

raw = """Size,radix_time,hash_time,radix_matches,hash_matches,radix_memMB,hash_memMB
10000,0.000211,0.000135,10000,10000,0.15,0.25
50000,0.000881,0.000785,50000,50000,0.76,1.00
100000,0.001895,0.001630,100000,100000,1.53,2.00
500000,0.008730,0.006774,500000,500000,7.65,8.00
1000000,0.014104,0.013434,1000000,1000000,15.29,16.00
5000000,0.060953,0.176781,5000000,5000000,76.54,128.00
10000000,0.107386,0.351367,10000000,10000000,153.09,256.00
20000000,0.224195,0.719218,20000000,20000000,306.18,512.00
50000000,0.533917,1.883415,50000000,50000000,764.94,1024.00
100000000,1.070534,3.788411,100000000,100000000,1529.88,2048.00
200000000,2.786814,7.638814,200000000,200000000,3059.76,4096.00
500000000,9.266365,20.765372,500000000,500000000,7645.39,8192.00"""

df = pd.read_csv(StringIO(raw))

# Plot 1: radix/hash time vs size
plt.figure(figsize=(8, 5))
plt.plot(df["Size"], df["radix_time"], marker="o", label="Radix hash join")
plt.plot(df["Size"], df["hash_time"], marker="o", label="Normal hash join")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Input size (tuples per table)")
plt.ylabel("Execution time (seconds)")
plt.title("Radix Hash Join vs Normal Hash Join: Time vs Size")
plt.legend()
plt.tight_layout()
time_plot = "./radix_vs_hash_time_vs_size.png"
plt.savefig(time_plot, dpi=200)
plt.close()

# Plot 2: radix/hash memory vs time
plt.figure(figsize=(8, 5))
plt.plot(df["Size"], df["radix_memMB"], marker="o", label="Radix hash join")
plt.plot(df["Size"], df["hash_memMB"], marker="o", label="Normal hash join")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Input size (tuples per table)")
plt.ylabel("Memory usage (MB)")
plt.title("Radix Hash Join vs Normal Hash Join: Memory vs Size")
plt.legend()
plt.tight_layout()
memory_plot = "./radix_vs_hash_memory_vs_size.png"
plt.savefig(memory_plot, dpi=200)
plt.close()

