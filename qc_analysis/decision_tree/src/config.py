import os
import json 

from typing import List, Dict, Union, Tuple, Iterable

from dataclasses import dataclass, field

@dataclass
class Config:
    """
    General configuration parser class.
    """
    directory: str = "config"
    filenames: List[str] = field(
        default_factory = lambda: ["settings.json"]
    )

    def __read_file(self, filename: str):
        """
        Read a single configuration file and yield a dictionary.
        """
        with open(os.path.join(self.directory, filename), "r") as f:
           yield json.load(f)
    
    def load(self) -> Dict[str, str]:
        """
        Read all files and yield dictionaries.
        """
        for file in self.filenames:
            yield from self.__read_file(file)

    