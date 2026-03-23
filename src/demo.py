import numpy as np
import matplotlib.pyplot as plt
import modes as ms

wavelengths = np.linspace(1.3, 1.6, 10)
n_LiTaO3 = [ms.materials.litao3(w, axis='o') for w in wavelengths]

plt.plot(wavelengths, n_LiTaO3)
plt.xlabel('wavelength (nm)')
plt.ylabel('Refractive index')
plt.title('LiTaO3 refractive index')
plt.show()
