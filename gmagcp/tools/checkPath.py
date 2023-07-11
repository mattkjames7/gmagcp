import os

def checkPath(path):
	
	if not os.path.isdir(path):
		os.makedirs(path)
