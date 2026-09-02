import sqlite3

import numpy as np

from second_brain.db import open_db
from second_brain.retrieval.semantic import SemanticIndex


class FakeEmbedder:
    model_id = "fake-openvino-embedding"
    available = True

    def embed(self, texts):
        vectors = []
        for text in texts:
            if "收敛" in text or "函数序列" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def test_embedding_model_id_versions_the_sampling_strategy(tmp_path):
    from second_brain.retrieval.semantic import OpenVINOEmbedder

    embedder = OpenVINOEmbedder(tmp_path / "embedding-model")
    assert embedder.model_id.endswith(":sample256:v1")


def test_semantic_paraphrase_recall_and_optional_absence(tmp_path):
    conn = open_db(tmp_path / "brain.db")
    document_id = conn.execute(
        "INSERT INTO documents(path,sha256,filename,file_type,title,status) VALUES('x','h','数学.docx','docx','数学','ready')"
    ).lastrowid
    conn.execute(
        "INSERT INTO chunks(document_id,block_index,chunk_index,source_kind,content,content_hash) VALUES(?,0,0,'text','一致收敛的判别方法','c')",
        (document_id,),
    )
    conn.commit()
    index = SemanticIndex(conn, FakeEmbedder())
    assert index.index_missing() == 1
    assert index.search("函数序列如何趋于同一极限")[0].filename == "数学.docx"
