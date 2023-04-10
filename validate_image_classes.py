import pathlib
import sys

if len(sys.argv) != 3:
  print()
  print('Checks all image annotation files (.txt) in the specified directory to make')
  print('sure that the class numbers match an expected value.  Annotations should be')
  print('in the yolo format with the class number as the first value on each line.')
  print()
  print(f'usage: {sys.argv[0]} <annotations directory> <class number>')
  print()
  exit(0)

found = False
dir = pathlib.Path(sys.argv[1])
annotation_files = [str(path) for path in dir.glob('*.txt')]
for filename in annotation_files:
  with open(filename) as f:
    for line in f:
      if line.split(' ')[0] != sys.argv[2]:
        if not found:
          print('The following have non-matching classes:')
        found = True
        print(f'{filename} ', end='')
if found:
  print()
else:
  print('All files have matching classes')

