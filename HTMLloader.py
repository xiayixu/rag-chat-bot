import logging
from pathlib import Path
from typing import Dict, Iterator, Union
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)

class BSHTMLLoader(BaseLoader):
    """Load `HTML` files and parse them with `beautiful soup`."""

    def __init__(
        self,
        file_path: Union[str, Path],
        open_encoding: Union[str, None] = None,
        bs_kwargs: Union[dict, None] = None,
        get_text_separator: str = "",
    ) -> None:
        """Initialize with path, and optionally, file encoding to use, and any kwargs
        to pass to the BeautifulSoup object.

        Args:
            file_path: The path to the file to load.
            open_encoding: The encoding to use when opening the file.
            bs_kwargs: Any kwargs to pass to the BeautifulSoup object.
            get_text_separator: The separator to use when calling get_text on the soup.
        """
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ImportError(
                "beautifulsoup4 package not found, please install it with "
                "`pip install beautifulsoup4`"
            )

        self.file_path = file_path
        self.open_encoding = open_encoding
        if bs_kwargs is None:
            bs_kwargs = {"features": "lxml"}
        self.bs_kwargs = bs_kwargs
        self.get_text_separator = get_text_separator

    def lazy_load(self) -> Iterator[Document]:
        """Load HTML document into document objects."""
        with open(self.file_path, "r", encoding=self.open_encoding) as f:
            soup = BeautifulSoup(f, **self.bs_kwargs)

        # Get all the text and handle tables
        for table in soup.find_all('table'):
            headers = [th.get_text(self.get_text_separator).strip() for th in table.find_all('tr')[0].find_all(['th', 'td'])]
            table_dicts = []
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                row_data = [col.get_text(self.get_text_separator).strip() for col in cols]
                # Create a dictionary for each row with header as keys
                row_dict = {header: value for header, value in zip(headers, row_data)}
                table_dicts.append(row_dict)
            
            # Format the dictionary list into a string
            table_text = "\n".join([", ".join(f"{k}: {v}" for k, v in row.items()) for row in table_dicts])
            table.replace_with(soup.new_string(table_text))

        # Extract all the text from the soup
        text = soup.get_text(self.get_text_separator)

        if soup.title:
            title = str(soup.title.string)
        else:
            title = ""

        meta_tag = soup.find("meta", attrs={"name": "ConfluencePageID"})
        if meta_tag:
            page_id = meta_tag.get("content", None)
        else:
            page_id = ''
        metadata: Dict[str, Union[str, None]] = {
            "source": str(self.file_path),
            "title": title,
            "page_id": page_id
        }
        yield Document(page_content=text, metadata=metadata)