import mongomock
import numpy as np
from typing import List, Dict, Any, Optional


class AsyncCursorWrapper:
    def __init__(self, cursor_data):
        self._cursor_data = cursor_data

    async def to_list(self, length=None):
        return list(self._cursor_data[:length])


class AsyncCollectionWrapper:
    def __init__(self):
        self._collection = mongomock.MongoClient().db.collection
        self._mocked_aggregate_result = None

    def find(self, *args, **kwargs):
        cursor = self._collection.find(*args, **kwargs)
        return AsyncCursorWrapper(cursor)

    async def insert_many(self, docs):
        return self._collection.insert_many(docs)

    def aggregate(self, pipeline, **kwargs):
        if self._mocked_aggregate_result is not None:
            return AsyncCursorWrapper(self._mocked_aggregate_result)

        # Check if this is a vector search pipeline
        if self._is_vector_search_pipeline(pipeline):
            return AsyncCursorWrapper(self._execute_vector_search(pipeline))

        # fallback: just return everything in the collection
        return AsyncCursorWrapper(self._collection.find())

    def mock_aggregate_result(self, docs):
        """Set custom results to be returned by aggregate()."""
        self._mocked_aggregate_result = docs

    def _is_vector_search_pipeline(self, pipeline: List[Dict]) -> bool:
        """Check if pipeline contains vector search stage"""
        return any("$vectorSearch" in stage for stage in pipeline)

    def _execute_vector_search(self, pipeline: List[Dict]) -> List[Dict]:
        """Execute vector search simulation with real similarity calculations"""

        # Extract vector search parameters
        vector_search_stage = None
        for stage in pipeline:
            if "$vectorSearch" in stage:
                vector_search_stage = stage["$vectorSearch"]
                break

        if not vector_search_stage:
            return []

        query_vector = np.array(vector_search_stage["queryVector"])
        vector_path = vector_search_stage["path"]
        limit = vector_search_stage.get("limit", 10)
        num_candidates = vector_search_stage.get("numCandidates", 100)

        # Get all documents from collection
        all_docs = list(self._collection.find())

        if not all_docs:
            return []

        # Calculate similarities
        similarities = []
        for doc in all_docs:
            # Extract vector from document using the path
            doc_vector = self._get_nested_value(doc, vector_path)
            if doc_vector is not None:
                similarity = self._cosine_similarity(query_vector, np.array(doc_vector))
                similarities.append((doc, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top candidates
        top_candidates = similarities[:min(num_candidates, len(similarities))]

        # Apply the rest of the pipeline stages
        results = []
        for doc, score in top_candidates:
            # Make a copy to avoid modifying original
            result_doc = doc.copy()
            results.append((result_doc, score))

        # Apply remaining pipeline stages
        return self._apply_remaining_stages(results, pipeline)

    def _get_nested_value(self, doc: Dict, path: str) -> Optional[List]:
        """Get nested value from document using dot notation path"""
        keys = path.split('.')
        current = doc

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current if isinstance(current, list) else None

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)

        # Clamp to valid range [-1, 1] to handle floating point precision issues
        return float(np.clip(similarity, -1.0, 1.0))

    def _apply_remaining_stages(self, results: List[tuple], pipeline: List[Dict]) -> List[Dict]:
        """Apply remaining pipeline stages after vector search"""
        current_results = []

        # Start processing from after $vectorSearch
        vector_search_index = 0
        for i, stage in enumerate(pipeline):
            if "$vectorSearch" in stage:
                vector_search_index = i + 1
                break

        # Add scores and convert to final format
        for doc, score in results:
            doc["score"] = score
            current_results.append(doc)

        # Apply remaining stages
        for stage in pipeline[vector_search_index:]:
            current_results = self._apply_stage(current_results, stage)

        return current_results

    def _apply_stage(self, docs: List[Dict], stage: Dict) -> List[Dict]:
        """Apply a single pipeline stage"""
        if "$addFields" in stage:
            # Handle $addFields for score (already added above)
            return docs

        elif "$match" in stage:
            # Apply match filters
            match_criteria = stage["$match"]
            filtered_docs = []

            for doc in docs:
                if self._document_matches(doc, match_criteria):
                    filtered_docs.append(doc)

            return filtered_docs

        elif "$project" in stage:
            # Apply projection
            projection = stage["$project"]
            projected_docs = []

            for doc in docs:
                projected_doc = {}
                for field, include in projection.items():
                    if include == 1 and field in doc:
                        projected_doc[field] = doc[field]
                projected_docs.append(projected_doc)

            return projected_docs

        # For unknown stages, just return docs unchanged
        return docs

    def _document_matches(self, doc: Dict, criteria: Dict) -> bool:
        """Check if document matches the given criteria"""
        for field, condition in criteria.items():
            doc_value = self._get_nested_value_for_match(doc, field)

            if isinstance(condition, dict):
                # Handle operators like $gte, $in
                for operator, value in condition.items():
                    if operator == "$gte":
                        if doc_value is None or doc_value < value:
                            return False
                    elif operator == "$in":
                        if doc_value not in value:
                            return False
                    # Add more operators as needed
            else:
                # Direct equality check
                if doc_value != condition:
                    return False

        return True

    def _get_nested_value_for_match(self, doc: Dict, path: str):
        """Get nested value for matching (similar to _get_nested_value but more flexible)"""
        keys = path.split('.')
        current = doc

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current


# Example usage:
if __name__ == "__main__":
    import asyncio


    async def test_vector_search():
        collection = AsyncCollectionWrapper()

        # Insert some test documents with vectors
        test_docs = [
            {
                "text": "Document about cats",
                "vector_field": [0.1, 0.2, 0.3, 0.4],
                "metadata": {"name_space": "animals"}
            },
            {
                "text": "Document about dogs",
                "vector_field": [0.2, 0.3, 0.4, 0.5],
                "metadata": {"name_space": "animals"}
            },
            {
                "text": "Document about cars",
                "vector_field": [0.8, 0.7, 0.6, 0.5],
                "metadata": {"name_space": "vehicles"}
            }
        ]

        await collection.insert_many(test_docs)

        # Test vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "vector_field",
                    "queryVector": [0.15, 0.25, 0.35, 0.45],  # Similar to first two docs
                    "numCandidates": 10,
                    "limit": 5,
                    "metric": "cosine"
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"score": {"$gte": 0.5}}},
            {
                "$project": {
                    "text": 1,
                    "metadata": 1,
                    "score": 1
                }
            }
        ]

        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list()

        print("Vector search results:")
        for result in results:
            print(f"Text: {result['text']}, Score: {result['score']:.3f}")


    # Run the test
    asyncio.run(test_vector_search())