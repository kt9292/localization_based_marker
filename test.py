import numpy as np

intrinsic_matrix = np.loadtxt("intrinsic_matrix.txt", dtype=float)
print(intrinsic_matrix)

intrinsic_matrix2 = np.array(((855.85858986, 0, 333.20121802),(0,856.38503924,216.40110069),(0,0,1)))

print(intrinsic_matrix2)

distortion_matrix = np.loadtxt("distortion_matrix.txt", dtype=float)
print(distortion_matrix)

distortion_matrix2 = np.array((0.08744123, -0.9721901, -0.00539752, 0.00223957, 2.18703157))
print(distortion_matrix2)