from byaldi import RAGMultiModalModel
import base64, io
from PIL import Image

RAG = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0", verbose=1)

def index_pdf(path: str):
    RAG.index(
        input_path=path,
        index_name="uploaded_doc",
        store_collection_with_index=True,
        overwrite=True
    )

def search_image(question: str):
    results = RAG.search(question, k=1)
    image_bytes = base64.b64decode(results[0].base64)
    return Image.open(io.BytesIO(image_bytes))
