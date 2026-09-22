from abstract_essentials import *
MARKS = ('"""', "'''")

def find_mark(line, mark, start=0):
  """Index of the next real `mark`, skipping one wrapped in quotes: sep = '\"\"\"'."""
  while True:
      i = line.find(mark, start)
      if i == -1:
          return -1
      before = line[i - 1] if i else ''
      after = line[i + 3] if i + 3 < len(line) else ''
      if before and before == after and before in '"\'':
          start = i + 4
          continue
      return i
      

def extract_quotes(file):
  quote_sections = []
  content = read_from_file(file)
  lines = content.split('\n')
  quotes = False          # are we inside a block right now
  is_doc = False          # did the block start its own line (a docstring)
  mark = '"""'
  current = []
  for line in lines:
      if not quotes: 
          stripped = eatAll(line, ['', ' ', '\t'])
          hits = [(find_mark(line, m), m) for m in MARKS]
          hits = [(i, m) for i, m in hits if i != -1]
          if not hits:    
              continue
          i, mark = min(hits)
          is_doc = stripped.startswith(mark) or \
              stripped.lstrip('rRbBuUfF').startswith(mark)
          rest = line[i + 3:] 
          j = find_mark(rest, mark)          
          if j != -1:                         # opens and closes on one line
              if is_doc:
                  quote_sections.append(rest[:j].strip())
          else:
              quotes = True                   # block stays open
              current = [rest]
      else: 
          j = find_mark(line, mark)
          if j == -1:
              current.append(line)
          else:
              current.append(line[:j])        # last line, up to the close
              if is_doc:
                  quote_sections.append('\n'.join(current).strip())
              quotes = False                  # closed again
              current = []
  return quote_sections



abs_file = os.path.abspath(__file__)
abs_dir = os.path.dirname(abs_file)
dir_list = os.listdir(abs_dir)
dirs,files = get_files_and_dirs(abs_dir,alowed_exts=[".py"])
for file in files:
    if file != abs_file:
        print(file)
        quotes = extract_quotes(file)
        for quote in quotes:
            input(quote)
