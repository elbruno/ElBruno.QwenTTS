using System.Numerics.Tensors;

namespace ElBruno.QwenTTS.Models;

/// <summary>
/// Top-K heap-based speaker similarity search using cosine similarity.
/// Finds the K most similar speaker embeddings from a reference collection.
/// Time complexity: O(n log k) vs O(n log n) for full sort.
/// </summary>
public sealed class SpeakerSimilaritySearch
{
    /// <summary>
    /// Finds the top K most similar speakers to a query embedding.
    /// </summary>
    /// <param name="queryEmbedding">Query speaker embedding (normalized or unnormalized).</param>
    /// <param name="referenceEmbeddings">Collection of reference embeddings with IDs.</param>
    /// <param name="k">Number of top results to return.</param>
    /// <returns>Top K matches ordered by similarity (highest first).</returns>
    public static SpeakerMatch[] FindTopK(ReadOnlySpan<float> queryEmbedding, IEnumerable<(string id, float[] embedding)> referenceEmbeddings, int k)
    {
        if (k <= 0)
            throw new ArgumentException("K must be positive", nameof(k));

        // Min-heap to track top K similarities
        var heap = new MinHeap(k);
        
        // Normalize query once
        Span<float> normalizedQuery = stackalloc float[queryEmbedding.Length];
        Normalize(queryEmbedding, normalizedQuery);

        Span<float> normalizedRef = stackalloc float[queryEmbedding.Length];
        
        foreach (var (id, embedding) in referenceEmbeddings)
        {
            if (embedding.Length != queryEmbedding.Length)
                throw new ArgumentException($"Embedding dimension mismatch: expected {queryEmbedding.Length}, got {embedding.Length}");

            // Normalize reference embedding
            Normalize(embedding, normalizedRef);
            
            // Compute cosine similarity using SIMD-accelerated dot product
            float similarity = TensorPrimitives.Dot(normalizedQuery, normalizedRef);
            
            // Insert into heap (only maintains top K)
            heap.Insert(new SpeakerMatch(id, similarity));
        }

        return heap.ExtractAll();
    }

    /// <summary>
    /// Normalizes a vector to unit length (L2 norm = 1).
    /// Uses SIMD-accelerated operations.
    /// </summary>
    private static void Normalize(ReadOnlySpan<float> input, Span<float> output)
    {
        float norm = TensorPrimitives.Norm(input);
        if (norm < 1e-8f)
        {
            // Zero vector — copy as-is
            input.CopyTo(output);
            return;
        }
        
        // output = input / norm
        TensorPrimitives.Divide(input, norm, output);
    }

    /// <summary>
    /// Min-heap that maintains the top K maximum values.
    /// When full, only accepts values larger than the minimum.
    /// </summary>
    private sealed class MinHeap
    {
        private readonly SpeakerMatch[] _heap;
        private int _size;
        private readonly int _capacity;

        public MinHeap(int capacity)
        {
            _capacity = capacity;
            _heap = new SpeakerMatch[capacity];
            _size = 0;
        }

        public void Insert(SpeakerMatch item)
        {
            if (_size < _capacity)
            {
                // Heap not full — insert and bubble up
                _heap[_size] = item;
                BubbleUp(_size);
                _size++;
            }
            else if (item.Similarity > _heap[0].Similarity)
            {
                // Replace minimum and bubble down
                _heap[0] = item;
                BubbleDown(0);
            }
        }

        public SpeakerMatch[] ExtractAll()
        {
            // Extract in descending order
            var result = new SpeakerMatch[_size];
            for (int i = _size - 1; i >= 0; i--)
            {
                result[i] = ExtractMin();
            }
            return result;
        }

        private SpeakerMatch ExtractMin()
        {
            var min = _heap[0];
            _size--;
            if (_size > 0)
            {
                _heap[0] = _heap[_size];
                BubbleDown(0);
            }
            return min;
        }

        private void BubbleUp(int index)
        {
            while (index > 0)
            {
                int parent = (index - 1) / 2;
                if (_heap[index].Similarity >= _heap[parent].Similarity)
                    break;
                
                (_heap[index], _heap[parent]) = (_heap[parent], _heap[index]);
                index = parent;
            }
        }

        private void BubbleDown(int index)
        {
            while (true)
            {
                int smallest = index;
                int left = 2 * index + 1;
                int right = 2 * index + 2;

                if (left < _size && _heap[left].Similarity < _heap[smallest].Similarity)
                    smallest = left;
                if (right < _size && _heap[right].Similarity < _heap[smallest].Similarity)
                    smallest = right;

                if (smallest == index)
                    break;

                (_heap[index], _heap[smallest]) = (_heap[smallest], _heap[index]);
                index = smallest;
            }
        }
    }
}

/// <summary>
/// Represents a speaker match result with ID and similarity score.
/// </summary>
public readonly record struct SpeakerMatch(string SpeakerId, float Similarity);
