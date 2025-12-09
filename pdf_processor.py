"""
PDF Ingestion & Preprocessing Module
Handles extraction of text, tables, code blocks, and images from PDFs
"""
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
import re
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from models import Chunk, ChunkType, PDFMetadata
from config import settings
import uuid


class PDFProcessor:
    """
    Comprehensive PDF processor that extracts:
    - Page-level text with character ranges
    - Tables (structured)
    - Code blocks (detected heuristically)
    - Images (with OCR)
    """

    def __init__(self):
        self.code_patterns = [
            re.compile(r'```[\s\S]*?```'),  # Markdown code blocks
            re.compile(r'def\s+\w+\s*\('),   # Python functions
            re.compile(r'class\s+\w+\s*[:\(]'),  # Python/Java classes
            re.compile(r'function\s+\w+\s*\('),  # JavaScript functions
            re.compile(r'public\s+class\s+\w+'),  # Java classes
        ]

    def process_pdf(self, filepath: str, pdf_id: str) -> Tuple[List[Chunk], PDFMetadata]:
        """
        Main entry point: process entire PDF and return chunks + metadata

        Args:
            filepath: Path to PDF file
            pdf_id: Unique identifier for this PDF

        Returns:
            Tuple of (chunks list, metadata object)
        """
        logger.info(f"Processing PDF: {filepath}")

        chunks: List[Chunk] = []

        # Open with PyMuPDF for text and images
        pdf_doc = fitz.open(filepath)
        num_pages = len(pdf_doc)

        # Open with pdfplumber for tables
        with pdfplumber.open(filepath) as plumber_pdf:
            for page_num in range(num_pages):
                logger.debug(f"Processing page {page_num + 1}/{num_pages}")

                # Extract from PyMuPDF
                fitz_page = pdf_doc[page_num]
                page_chunks = self._process_page(
                    fitz_page=fitz_page,
                    plumber_page=plumber_pdf.pages[page_num],
                    page_num=page_num + 1,  # 1-indexed
                    pdf_id=pdf_id
                )
                chunks.extend(page_chunks)

        pdf_doc.close()

        # Create metadata
        import os
        file_size = os.path.getsize(filepath)
        metadata = PDFMetadata(
            pdf_id=pdf_id,
            filename=os.path.basename(filepath),
            filepath=filepath,
            num_pages=num_pages,
            file_size_bytes=file_size,
            num_chunks=len(chunks),
            processing_status="completed"
        )

        logger.info(f"Extracted {len(chunks)} chunks from {num_pages} pages")
        return chunks, metadata

    def _process_page(
        self,
        fitz_page,
        plumber_page,
        page_num: int,
        pdf_id: str
    ) -> List[Chunk]:
        """Process a single page and return all chunks"""
        chunks: List[Chunk] = []

        # 1. Extract raw text with character positions
        page_text = fitz_page.get_text("text")

        # 2. Extract tables
        table_chunks = self._extract_tables(plumber_page, page_num, pdf_id)
        chunks.extend(table_chunks)

        # 3. Extract code blocks
        code_chunks = self._extract_code_blocks(page_text, page_num, pdf_id)
        chunks.extend(code_chunks)

        # 4. Extract images and run OCR
        image_chunks = self._extract_images(fitz_page, page_num, pdf_id)
        chunks.extend(image_chunks)

        # 5. Extract remaining text as paragraphs
        # Remove table and code regions from text before creating paragraph chunks
        cleaned_text = self._remove_extracted_regions(
            page_text,
            [c.text for c in code_chunks]
        )

        if cleaned_text.strip():
            para_chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                pdf_id=pdf_id,
                page_number=page_num,
                char_range=(0, len(cleaned_text)),
                type=ChunkType.PARAGRAPH,
                text=cleaned_text,
                metadata={"source": "text_extraction"}
            )
            chunks.append(para_chunk)

        return chunks

    def _extract_tables(self, plumber_page, page_num: int, pdf_id: str) -> List[Chunk]:
        """Extract tables from page using pdfplumber"""
        chunks = []
        tables = plumber_page.extract_tables()

        for idx, table in enumerate(tables):
            if not table:
                continue

            # Convert table to structured JSON
            table_json = self._table_to_json(table)

            # Convert table to text representation
            table_text = self._table_to_text(table)

            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                pdf_id=pdf_id,
                page_number=page_num,
                char_range=(0, len(table_text)),
                type=ChunkType.TABLE,
                text=table_text,
                table_json=table_json,
                metadata={"table_index": idx, "num_rows": len(table)}
            )
            chunks.append(chunk)

        logger.debug(f"Extracted {len(chunks)} tables from page {page_num}")
        return chunks

    def _table_to_json(self, table: List[List[str]]) -> Dict[str, Any]:
        """Convert table to structured JSON"""
        if not table or len(table) < 2:
            return {"headers": [], "rows": []}

        headers = table[0]
        rows = table[1:]

        return {
            "headers": headers,
            "rows": [
                {headers[i]: cell for i, cell in enumerate(row) if i < len(headers)}
                for row in rows
            ]
        }

    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table to readable text"""
        return "\n".join([" | ".join([str(cell) for cell in row]) for row in table])

    def _extract_code_blocks(self, text: str, page_num: int, pdf_id: str) -> List[Chunk]:
        """Extract code blocks using heuristic patterns"""
        chunks = []

        # Look for code patterns
        for pattern in self.code_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                code_text = match.group(0)
                if len(code_text) < 20:  # Skip very short matches
                    continue

                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    pdf_id=pdf_id,
                    page_number=page_num,
                    char_range=(match.start(), match.end()),
                    type=ChunkType.CODE,
                    text=code_text,
                    metadata={
                        "pattern": pattern.pattern,
                        "detected_language": self._detect_language(code_text)
                    }
                )
                chunks.append(chunk)

        # Also detect monospace font regions (if PDF has font info)
        # This is more advanced and would require font analysis

        logger.debug(f"Extracted {len(chunks)} code blocks from page {page_num}")
        return chunks

    def _detect_language(self, code: str) -> str:
        """Heuristically detect programming language"""
        if 'def ' in code and ':' in code:
            return 'python'
        elif 'function' in code or 'const' in code or 'let' in code:
            return 'javascript'
        elif 'public class' in code or 'private' in code:
            return 'java'
        elif '#include' in code:
            return 'c++'
        else:
            return 'unknown'

    def _extract_images(self, fitz_page, page_num: int, pdf_id: str) -> List[Chunk]:
        """Extract images and run OCR"""
        chunks = []
        image_list = fitz_page.get_images()

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = fitz_page.parent.extract_image(xref)
                image_bytes = base_image["image"]

                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes))

                # Run OCR
                ocr_text = pytesseract.image_to_string(image)

                if ocr_text.strip():
                    image_id = f"{pdf_id}_p{page_num}_img{img_index}"

                    chunk = Chunk(
                        chunk_id=str(uuid.uuid4()),
                        pdf_id=pdf_id,
                        page_number=page_num,
                        char_range=(0, len(ocr_text)),
                        type=ChunkType.IMAGE_TEXT,
                        text=ocr_text,
                        image_id=image_id,
                        metadata={
                            "image_format": base_image["ext"],
                            "image_index": img_index
                        }
                    )
                    chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Failed to extract image {img_index} on page {page_num}: {e}")

        logger.debug(f"Extracted {len(chunks)} images from page {page_num}")
        return chunks

    def _remove_extracted_regions(self, text: str, code_blocks: List[str]) -> str:
        """Remove already-extracted code blocks from text"""
        for code in code_blocks:
            text = text.replace(code, "")
        return text

    def chunk_text(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Further chunk large text blocks into smaller overlapping chunks

        Args:
            chunks: Initial chunks from PDF extraction

        Returns:
            Refined chunks with proper overlap
        """
        refined_chunks = []

        for chunk in chunks:
            # Skip non-text chunks (tables, images already chunked)
            if chunk.type in [ChunkType.TABLE, ChunkType.CODE]:
                refined_chunks.append(chunk)
                continue

            # Split long paragraphs into smaller chunks with overlap
            text = chunk.text
            chunk_size = settings.chunk_size
            overlap = settings.chunk_overlap

            if len(text) <= chunk_size:
                refined_chunks.append(chunk)
                continue

            # Create overlapping windows
            for i in range(0, len(text), chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]

                if len(chunk_text) < settings.min_chunk_size:
                    continue

                new_chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    pdf_id=chunk.pdf_id,
                    page_number=chunk.page_number,
                    char_range=(i, i + len(chunk_text)),
                    type=chunk.type,
                    text=chunk_text,
                    metadata={
                        **chunk.metadata,
                        "parent_chunk_id": chunk.chunk_id,
                        "window_index": i // (chunk_size - overlap)
                    }
                )
                refined_chunks.append(new_chunk)

        logger.info(f"Refined {len(chunks)} chunks into {len(refined_chunks)} chunks")
        return refined_chunks
