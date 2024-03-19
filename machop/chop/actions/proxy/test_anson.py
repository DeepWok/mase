import os


current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))


print(parent_directory)
