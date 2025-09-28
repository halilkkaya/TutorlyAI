"""
Large File Processing Optimization
Memory-efficient PDF processing, streaming, chunked operations
"""

import logging
import asyncio
import io
import os
from typing import AsyncGenerator, List, Dict, Any, Optional
from pathlib import Path
import aiofiles
from PyPDF2 import PdfReader
import gc
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FileProcessingConfig:
    max_file_size_mb: int = 50
    chunk_size_bytes: int = 1024 * 1024  # 1MB chunks
    max_pages_per_batch: int = 10
    memory_threshold_mb: int = 100
    enable_garbage_collection: bool = True

class MemoryEfficientPDFProcessor:
    def __init__(self, config: FileProcessingConfig = None):
        self.config = config or FileProcessingConfig()
        self.stats = {
            "files_processed": 0,
            "pages_processed": 0,
            "memory_cleanups": 0,
            "processing_errors": 0
        }

    async def process_pdf_async(self, pdf_path: str, metadata: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Memory-efficient async PDF processing
        Processes PDF in batches to avoid memory issues
        """
        try:
            # File size kontrolü
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                logger.warning(f"[FILE] Large file detected: {file_size_mb:.1f}MB - {pdf_path}")

            logger.info(f"[FILE] Processing PDF: {Path(pdf_path).name} ({file_size_mb:.1f}MB)")

            # PDF'i memory-mapped okuma ile aç
            async with aiofiles.open(pdf_path, 'rb') as file:
                pdf_data = await file.read()

            # PDF reader oluştur
            pdf_reader = PdfReader(io.BytesIO(pdf_data))
            total_pages = len(pdf_reader.pages)

            logger.info(f"[FILE] {total_pages} pages to process")

            # Sayfaları batch'ler halinde işle
            for batch_start in range(0, total_pages, self.config.max_pages_per_batch):
                batch_end = min(batch_start + self.config.max_pages_per_batch, total_pages)
                logger.debug(f"[FILE] Processing pages {batch_start}-{batch_end}")

                batch_text = ""
                pages_in_batch = 0

                for page_num in range(batch_start, batch_end):
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()

                        if text and text.strip():
                            # Metin temizleme
                            cleaned_text = self._clean_text(text)
                            batch_text += cleaned_text + "\n"
                            pages_in_batch += 1

                        self.stats["pages_processed"] += 1

                        # Memory kontrolü
                        if await self._check_memory_usage():
                            logger.info("[FILE] High memory usage detected, forcing GC")
                            await self._force_garbage_collection()

                    except Exception as e:
                        logger.error(f"[FILE] Error processing page {page_num}: {str(e)}")
                        self.stats["processing_errors"] += 1
                        continue

                # Batch'i yield et
                if batch_text.strip():
                    yield {
                        "text": batch_text,
                        "metadata": {
                            **metadata,
                            "batch_start": batch_start,
                            "batch_end": batch_end,
                            "pages_in_batch": pages_in_batch,
                            "total_pages": total_pages
                        }
                    }

                # Batch sonrası memory temizleme
                if self.config.enable_garbage_collection:
                    await asyncio.sleep(0.01)  # Çok kısa yield
                    gc.collect()

            self.stats["files_processed"] += 1
            logger.info(f"[FILE] Completed processing: {Path(pdf_path).name}")

        except Exception as e:
            logger.error(f"[FILE] Error processing PDF {pdf_path}: {str(e)}")
            self.stats["processing_errors"] += 1
            raise

    def _clean_text(self, text: str) -> str:
        """Metin temizleme optimized"""
        import re

        if not text:
            return ""

        # Whitespace normalizasyonu
        text = re.sub(r'\s+', ' ', text)

        # Özel karakterleri temizle
        text = re.sub(r'[^\w\s\u00C0-\u017F]', ' ', text)

        return text.strip()

    async def _check_memory_usage(self) -> bool:
        """Memory kullanımını kontrol et"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb > self.config.memory_threshold_mb
        except ImportError:
            return False

    async def _force_garbage_collection(self):
        """Zorla garbage collection"""
        gc.collect()
        self.stats["memory_cleanups"] += 1
        await asyncio.sleep(0.1)  # GC'nin tamamlanması için kısa bekleme

    def get_stats(self) -> Dict[str, Any]:
        """Processing istatistikleri"""
        return {
            **self.stats,
            "config": {
                "max_file_size_mb": self.config.max_file_size_mb,
                "chunk_size_bytes": self.config.chunk_size_bytes,
                "max_pages_per_batch": self.config.max_pages_per_batch,
                "memory_threshold_mb": self.config.memory_threshold_mb
            }
        }

# =========================
# STREAMING FILE OPERATIONS
# =========================

class StreamingFileHandler:
    def __init__(self, config: FileProcessingConfig = None):
        self.config = config or FileProcessingConfig()

    async def stream_large_file(self, file_path: str) -> AsyncGenerator[bytes, None]:
        """
        Large file'ları chunk'lar halinde stream et
        """
        try:
            async with aiofiles.open(file_path, 'rb') as file:
                while True:
                    chunk = await file.read(self.config.chunk_size_bytes)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            logger.error(f"[STREAM] Error streaming file {file_path}: {str(e)}")
            raise

    async def process_file_chunks(self, file_path: str, processor_func) -> List[Any]:
        """
        File'ı chunk'lar halinde işle
        """
        results = []
        chunk_num = 0

        async for chunk in self.stream_large_file(file_path):
            try:
                result = await processor_func(chunk, chunk_num)
                if result:
                    results.append(result)
                chunk_num += 1
            except Exception as e:
                logger.error(f"[STREAM] Error processing chunk {chunk_num}: {str(e)}")
                continue

        return results

# =========================
# GLOBAL INSTANCES
# =========================

file_config = FileProcessingConfig()
pdf_processor = MemoryEfficientPDFProcessor(file_config)
stream_handler = StreamingFileHandler(file_config)

def get_file_processing_stats() -> Dict[str, Any]:
    """File processing istatistikleri"""
    return {
        "pdf_processor": pdf_processor.get_stats(),
        "config": {
            "max_file_size_mb": file_config.max_file_size_mb,
            "chunk_size_bytes": file_config.chunk_size_bytes,
            "max_pages_per_batch": file_config.max_pages_per_batch
        }
    }