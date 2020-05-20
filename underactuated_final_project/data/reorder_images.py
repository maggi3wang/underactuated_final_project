import os

package_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(package_dir, '../data/all/images')
files = sorted(os.listdir(image_dir))
#print(files)

for index, file in enumerate(files):
	os.rename(os.path.join(image_dir, file), os.path.join(image_dir, '{:06d}.png'.format(index)))