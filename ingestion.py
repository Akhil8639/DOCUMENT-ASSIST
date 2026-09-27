import asyncio
import os

import certifi
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilyCrawl

from logger import (
    Colors,
    log_error,
    log_header,
    log_info,
    log_success,
    log_warning,
)


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ============================================================
# 2. CHECK API KEYS
# ============================================================

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY is missing from your .env file"
    )

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError(
        "TAVILY_API_KEY is missing from your .env file"
    )


# ============================================================
# 3. CREATE GEMINI EMBEDDING MODEL
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=1536,
)


# ============================================================
# 4. CREATE CHROMA VECTOR STORE
# ============================================================

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)


# ============================================================
# 5. CREATE TAVILY CRAWLER
# ============================================================

tavily_crawl = TavilyCrawl()


# ============================================================
# 6. STORE DOCUMENTS IN CHROMA
# ============================================================

async def index_documents_async(
    documents: list[Document],
    batch_size: int = 10,
    delay_between_batches: int = 5,
    max_retries: int = 3,
):
    """
    Divide document chunks into smaller batches
    and store them sequentially in ChromaDB.

    Retries when Gemini returns rate-limit errors.
    """

    log_header("VECTOR STORAGE PHASE")

    log_info(
        f"Preparing to store {len(documents)} chunks",
        Colors.DARKCYAN,
    )

    # --------------------------------------------------------
    # Create smaller batches
    # --------------------------------------------------------

    batches = [
        documents[i:i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"Created {len(batches)} batches "
        f"with maximum {batch_size} chunks per batch"
    )

    # --------------------------------------------------------
    # Function to store ONE batch
    # --------------------------------------------------------

    async def add_batch(
        batch: list[Document],
        batch_number: int,
    ):
        for attempt in range(1, max_retries + 1):

            try:
                log_info(
                    f"Processing batch "
                    f"{batch_number}/{len(batches)} "
                    f"({len(batch)} chunks) "
                    f"- Attempt {attempt}/{max_retries}"
                )

                # Gemini creates embeddings.
                # Chroma stores embeddings + text + metadata.
                await vectorstore.aadd_documents(batch)

                log_success(
                    f"Batch {batch_number}/{len(batches)} "
                    f"stored successfully "
                    f"({len(batch)} chunks)"
                )

                return True

            except Exception as e:

                error_message = str(e)

                # --------------------------------------------
                # Gemini rate-limit / quota error
                # --------------------------------------------

                if (
                    "429" in error_message
                    or "RESOURCE_EXHAUSTED" in error_message
                ):

                    log_warning(
                        f"Gemini rate limit reached "
                        f"for batch {batch_number}"
                    )

                    if attempt < max_retries:

                        # Increase wait time after each retry
                        wait_time = 10 * attempt

                        log_info(
                            f"Waiting {wait_time} seconds "
                            f"before retrying...",
                            Colors.YELLOW,
                        )

                        await asyncio.sleep(wait_time)

                    else:

                        log_error(
                            f"Batch {batch_number} failed "
                            f"after {max_retries} attempts"
                        )

                        return False

                else:

                    log_error(
                        f"Batch {batch_number} failed: {e}"
                    )

                    return False

        return False

    # --------------------------------------------------------
    # Process batches SEQUENTIALLY
    # --------------------------------------------------------

    results = []

    for i, batch in enumerate(batches):

        batch_number = i + 1

        result = await add_batch(
            batch,
            batch_number,
        )

        results.append(result)

        # Stop immediately if a batch completely fails.
        # This avoids pretending the DB is fully indexed.
        if result is False:

            log_error(
                f"Stopping indexing because "
                f"batch {batch_number} failed."
            )

            return False

        # ----------------------------------------------------
        # Delay before sending next batch to Gemini
        # ----------------------------------------------------

        if batch_number < len(batches):

            log_info(
                f"Waiting {delay_between_batches} seconds "
                f"before next batch...",
                Colors.YELLOW,
            )

            await asyncio.sleep(
                delay_between_batches
            )

    # --------------------------------------------------------
    # Count successful batches
    # --------------------------------------------------------

    successful_batches = sum(
        1
        for result in results
        if result is True
    )

    if successful_batches == len(batches):

        log_success(
            f"All batches stored successfully! "
            f"({successful_batches}/{len(batches)})"
        )

        return True

    log_warning(
        f"Only {successful_batches}/{len(batches)} "
        f"batches were stored successfully"
    )

    return False


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

async def main():

    log_header(
        "DOCUMENTATION INGESTION PIPELINE"
    )

    # ========================================================
    # STEP 1: CRAWL LANGCHAIN DOCUMENTATION
    # ========================================================

    log_info(
        "TavilyCrawl: Starting LangChain documentation crawl",
        Colors.PURPLE,
    )

    try:

        response = tavily_crawl.invoke(
            {
                "url": "https://python.langchain.com/",
                "max_depth": 2,
                "extract_depth": "advanced",
            }
        )

        log_success(
            "TavilyCrawl: Website crawling completed"
        )

    except Exception as e:

        log_error(
            f"TavilyCrawl failed: {e}"
        )

        return


    # ========================================================
    # STEP 2: CREATE LANGCHAIN DOCUMENTS
    # ========================================================

    log_header(
        "DOCUMENT CREATION PHASE"
    )

    all_docs = []

    results = response.get(
        "results",
        [],
    )

    log_info(
        f"Tavily returned "
        f"{len(results)} webpage results"
    )

    for item in results:

        raw_content = item.get(
            "raw_content"
        )

        url = item.get(
            "url"
        )

        # --------------------------------------------
        # Skip empty pages
        # --------------------------------------------

        if not raw_content:

            log_warning(
                f"Skipping empty page: {url}"
            )

            continue

        # --------------------------------------------
        # Create LangChain Document
        # --------------------------------------------

        document = Document(
            page_content=raw_content,
            metadata={
                "source": url
            },
        )

        all_docs.append(
            document
        )

        log_info(
            f"Created Document from: {url}",
            Colors.DARKCYAN,
        )

    # --------------------------------------------------------
    # Stop if no documents were created
    # --------------------------------------------------------

    if not all_docs:

        log_error(
            "No documents were created."
        )

        return

    log_success(
        f"Created {len(all_docs)} "
        f"LangChain Documents"
    )


    # ========================================================
    # STEP 3: SPLIT DOCUMENTS INTO CHUNKS
    # ========================================================

    log_header(
        "DOCUMENT CHUNKING PHASE"
    )

    log_info(
        f"Splitting {len(all_docs)} documents "
        f"using chunk_size=4000 "
        f"and chunk_overlap=200",
        Colors.YELLOW,
    )

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
        )
    )

    chunks = (
        text_splitter.split_documents(
            all_docs
        )
    )

    log_success(
        f"Created {len(chunks)} chunks "
        f"from {len(all_docs)} documents"
    )


    # ========================================================
    # STEP 4: EMBEDDINGS + CHROMA STORAGE
    # ========================================================

    log_header(
        "EMBEDDING AND STORAGE PHASE"
    )

    log_info(
        "Using Gemini embedding model: "
        "gemini-embedding-2",
        Colors.PURPLE,
    )

    log_info(
        "Embedding dimension: 1536"
    )

    log_info(
        "Using sequential batches to reduce "
        "Gemini rate-limit errors"
    )

    indexing_success = (
        await index_documents_async(
            chunks,
            batch_size=10,
            delay_between_batches=5,
            max_retries=3,
        )
    )

    # --------------------------------------------------------
    # Stop if vector indexing failed
    # --------------------------------------------------------

    if not indexing_success:

        log_header(
            "PIPELINE FAILED"
        )

        log_error(
            "Document crawling and chunking worked, "
            "but vector indexing did not complete."
        )

        log_warning(
            "Check your Gemini API quota/rate limits "
            "and try again later."
        )

        return


    # ========================================================
    # PIPELINE COMPLETE
    # ========================================================

    log_header(
        "PIPELINE COMPLETE"
    )

    log_success(
        "Documentation ingestion pipeline "
        "finished successfully!"
    )

    log_info(
        "SUMMARY",
        Colors.BOLD,
    )

    log_info(
        f"Documents extracted : "
        f"{len(all_docs)}"
    )

    log_info(
        f"Chunks created      : "
        f"{len(chunks)}"
    )

    log_info(
        "Embedding model     : "
        "gemini-embedding-2"
    )

    log_info(
        "Embedding dimension : "
        "1536"
    )

    log_info(
        "Vector database     : "
        "ChromaDB"
    )

    log_info(
        "Persist directory   : "
        "chroma_db"
    )


# ============================================================
# 8. RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )