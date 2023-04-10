import pathlib
import sys
import shutil

if len(sys.argv) != 4:
  print()
  print('Moves a specified percentage of image files (.jpg or .png) as well as')
  print('their corresponding annotation files (.txt) from a training directory')
  print('to a validation directory.')
  print()
  print(f'usage: {sys.argv[0]} <train dir> <val dir> <val percent>')
  print()
  exit(0)

dir = pathlib.Path(sys.argv[1])
total_file_count = len([f for f in dir.glob('*.*')])
print(f'total files: {total_file_count}')
annotation_files = [str(path) for path in dir.glob('*.txt')]
image_files = [str(path) for path in dir.glob('*.[jp][pn]g')]
print(f'image files: {len(image_files)}')
print(f'annotation files: {len(annotation_files)}')

if len(annotation_files) != len(image_files):
  print('file count mismatch--must have exactly one annotation file per image file')
  exit(0)

move_pct_raw = float(sys.argv[3])
move_pct = move_pct_raw if move_pct_raw < 1.0 else move_pct_raw / 100.0
move_count = int((total_file_count / 2) * move_pct)

moved = {'image':0, 'annot':0}
for file in image_files[:move_count]:
    shutil.move(file, sys.argv[2])
    moved['image'] += 1
    file = file[:-3] + 'txt'
    shutil.move(file, sys.argv[2])
    moved['annot'] += 1

print(f'moved {moved["image"]} image files and {moved["annot"]} annotation files to {sys.argv[2]}')
