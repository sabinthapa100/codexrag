from .python_ast_parser import PythonASTParser
from .notebook_parser import NotebookParser
from .pdf_parser import PDFParser
from .csv_parser import CSVParser
from .text_like import TextLikeParser
from .cpp_block_parser import CPPBlockParser

DEFAULT_PARSERS = [
    PythonASTParser(),
    NotebookParser(),
    CPPBlockParser(),
    PDFParser(),
    CSVParser(),
    TextLikeParser(),
]
