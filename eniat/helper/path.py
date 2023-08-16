from pathlib import Path
import os
import re


class DirTree(object):
    """
    took from https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    answered by user: abstrus

    Examples:
        paths = DirTree.make_tree(path)
        for p in paths:
            print(p.displayable())
    """
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if cls._default_criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return isinstance(path, str)

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))



class PathMan:
    '''
    Path manager v0.0.1a (alpha)
    Author: SungHo Lee(shlee@unc.edu)
    '''
    def __init__(self, path, regex=None, ext=None):
        self._path = path
        self._prep_path()
        self.set_filters(regex=regex, ext=ext)
    
    def _prep_path(self):
        'check if the initiated path exists, mkdir if not.'
        if not os.path.exists(self._path):
            os.mkdir(self._path)
            
    def set_filters(self, regex=None, ext=None):
        self._regex = regex
        self._ext = ext
    
    @property
    def is_filtered(self):
        'return True if filters applied'
        if self._regex is not None or self._ext is not None:
            return True
        else:
            return False
            
    def listdir(self):
        '''
        list files
        regex(str): regular express
        ext(str): file extention
        '''
        result = [p for p in os.listdir(self._path)]
        if self._regex is not None:
            result = [p for p in result if re.match(self._regex, p)]
        if self._ext is not None:
            result = [p for p in result if re.match(fr'.*\.{self._ext}', p)]
        return {i:p for i, p in enumerate(result)}
    
    def isdir(self):
        return {i:(p, os.path.isdir(self(p))) for i, p in self.listdir().items()}
    
    def isfile(self):
        return {i:(p, os.path.isfile(self(p))) for i, p in self.listdir().items()}
    
    def chdir(self, index):
        if self.isdir()[index][0]:
            return PathMan(self[index])
        else:
            raise Exception(f'{self[index]} is not a directory.')
            
    def mkdir(self, path):
        'make dir if not exists then return correspond PathMan object'
        if os.path.exists(self(path)):
            raise Exception(f'{self(path)} exists')
        else:
            os.mkdir(self(path))
        return PathMan(self(path))
            
    def __call__(self, fname):
        'return joined path'
        return os.path.join(self._path, fname)
    
    def __len__(self):
        return len(self.listdir())
                      
    def __getitem__(self, index):
        'indexing'
        return self(self.listdir()[index])
                      
    def __iter__(self):
        for idx, f in self.listdir().items():
            yield idx, self(f)
            
    def __repr__(self):
        message = [f"PathMan object ({os.path.abspath(self._path)})"]
        if self.is_filtered:
            message.append(f" -Filtered applied")
            if self._regex is not None:
                message.append(f"\tregex: '{self._regex}'")
            if self._ext is not None:
                message.append(f"\text: '{self._ext}'")
        message.append(f" -Number of files: {len(self)}")
        message.append(f"\tFiles: {sum([int(b) for p, b in self.isfile().values()])}")
        message.append(f"\tDirectories: {sum([int(b) for p, b in self.isdir().values()])}")
        
        return '\n'.join(message)