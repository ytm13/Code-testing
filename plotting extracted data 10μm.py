import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("rnr 10micron alumina.csv", header=1)

plt.figure(figsize=(8,5))

plt.scatter(data["X9"], data["Y9"], marker="o", label="RUN9")
plt.scatter(data["X10"], data["Y10"], marker="s", label="RUN10")
plt.scatter(data["X15"], data["Y15"], marker="^", label="RUN15")

plt.title("10μm alumina particles")
plt.xlabel("friction velocity (m/s)")
plt.ylabel("remainded fraction")
plt.xscale("log")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()