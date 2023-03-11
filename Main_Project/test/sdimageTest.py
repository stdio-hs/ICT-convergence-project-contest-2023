from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ascent = misc.face()
result = ndimage.maximum_filter(ascent, size=20, mode='constant')
print(ascent)
print('result : ', result)

ax1.imshow(ascent)
ax2.imshow(result)
plt.show()