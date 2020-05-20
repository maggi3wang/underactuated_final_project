from ..model.model import Model
import os

def main():
	package_dir = os.path.dirname(os.path.abspath(__file__))
	train_dir = os.path.join(package_dir, '../data/train')
	test_dir = os.path.join(package_dir, '../data/test')
	models_dir = os.path.join(package_dir, 'models')

	global_net = Model(train_dir=train_dir,
			  		   test_dir=test_dir,
			  		   models_dir=models_dir)

	global_net.train(num_epochs=80, siamese=True)

if __name__ == "__main__":
    main()

