import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("rnr 20micron alumina.csv", header=1)

plt.figure(figsize=(8,5))

plt.scatter(data["X7"], data["Y7"], marker="o", label="RUN7")
plt.scatter(data["X8"], data["Y8"], marker="s", label="RUN8")
plt.scatter(data["X20"], data["Y20"], marker="^", label="RUN20")

plt.title("20μm alumina particles")
plt.xlabel("friction velocity (m/s)")
plt.ylabel("remainded fraction")
plt.xscale("log")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()