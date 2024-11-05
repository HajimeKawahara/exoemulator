import matplotlib.pyplot as plt
import numpy as np

tag = "decoder_3lrc"
#tag = "Pattmp_n100niter1500000"

arr = np.load("loss"+tag+".npz")
lossarr = arr["lossarr"]
testlossarr = arr["testlossarr"]

plt.plot(lossarr[10:], label="train",alpha=0.1)
plt.plot(testlossarr[10:], label="test",alpha=0.1)
plt.yscale("log")
#plt.xscale("log")
plt.legend()
plt.savefig("loss_"+tag+".png")
