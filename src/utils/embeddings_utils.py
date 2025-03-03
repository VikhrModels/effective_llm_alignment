import torch
import numpy as np
import faiss
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm


def shuffle_matrix_with_mapping(matrix: np.ndarray):
    N = matrix.shape[0]
    permuted_indices = np.random.permutation(N)
    shuffled_matrix = matrix[permuted_indices]
    return shuffled_matrix, permuted_indices


def normalize_embeddings(embeddings: np.array):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def cosine_similarity_matrix(embeddings: np.ndarray):
    normalized_embeddings = normalize_embeddings(embeddings)
    cosine_similarity = np.dot(normalized_embeddings, normalized_embeddings.T)
    return cosine_similarity


def _average_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def process_texts(texts, batch_size, model, tokenizer, device, normalize=True):
    embeddings = []
    texts = list(texts)

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        batch = tokenizer.batch_encode_plus(
            batch_texts, return_tensors="pt", padding=True, truncation=True
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            output = model(**batch).last_hidden_state
            pooled_output = _average_pool(output, batch["attention_mask"])
        embeddings.append(pooled_output.detach().cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()
    if normalize:
        embeddings = normalize_embeddings(embeddings)
    return embeddings


def _faiss_deduplicate_single_v1(
    embeddings: np.ndarray,
    similarity_threshold=0.9,
    use_sqrt_neibs=False,
    index_class=faiss.IndexFlatIP,
    index_args=[],
):
    # Initialize FAISS index
    index = index_class(
        embeddings.shape[1], *index_args
    )  # Use inner product for similarity measure

    # Add embeddings to index
    index.add(embeddings)

    # Initialize an array to keep track of embeddings to keep
    keep = np.ones(len(embeddings), dtype=bool)

    for i in range(len(embeddings)):
        if keep[i]:
            # Search for nearest neighbors
            D, I = index.search(
                embeddings[i : i + 1],
                len(embeddings)
                if not use_sqrt_neibs
                else int(np.sqrt(len(embeddings))),
            )
            # Since embeddings are normalized, similarities are between -1 and 1
            # Find duplicates with similarity above the threshold (excluding self)
            duplicates = I[0][(D[0] > similarity_threshold) & (I[0] != i)]
            # Mark duplicates for removal
            keep[duplicates] = False

    # Get indices of unique embeddings to keep
    unique_indices = np.where(keep)[0]
    return embeddings[unique_indices], unique_indices


def _faiss_deduplicate_single_v2(
    embeddings: np.ndarray,
    similarity_threshold=0.9,
    use_sqrt_neibs=False,
    index_class=faiss.IndexFlatIP,
    index_args=[],
):
    # Initialize FAISS index (inner product for similarity measure)
    index = index_class(embeddings.shape[1], *index_args)
    index.add(embeddings)

    # Number of neighbors to search for (all embeddings by default, or sqrt if use_sqrt_neibs is True)
    num_neibs = len(embeddings) if not use_sqrt_neibs else int(np.sqrt(len(embeddings)))

    # Perform a batch search to retrieve all neighbors
    D, I = index.search(embeddings, num_neibs)

    # Since cosine similarity is used, filter out neighbors based on the similarity threshold
    # and remove all self-matches (where I[i] == i)
    mask_self = np.arange(len(embeddings))[:, None] != I  # Exclude the self-match index
    mask_similar = D > similarity_threshold  # Apply similarity threshold
    final_mask = mask_self & mask_similar

    # Create an array that tracks whether to keep an embedding (True) or not (False)
    discard = np.zeros(len(embeddings), dtype=bool)

    # Mark embeddings for removal based on the first occurrence of duplicates
    for i in range(len(embeddings)):
        if not discard[
            i
        ]:  # We only consider embeddings that haven't been marked for removal yet
            duplicates = I[i][final_mask[i]]  # Get the current duplicates
            discard[duplicates] = True  # Mark duplicates for removal

    # Inverse the discard mask to get the keep indices
    keep = ~discard

    # Return the deduplicated embeddings and their indices
    unique_indices = np.where(keep)[0]
    return embeddings[unique_indices], unique_indices


def _faiss_deduplicate_single_v3(
    embeddings: np.ndarray,
    similarity_threshold=0.9,
    index_class=faiss.IndexFlatIP,
    index_args=[],
):
    # Initialize FAISS index (with inner product similarity by default)
    index = index_class(embeddings.shape[1], *index_args)

    # Add embeddings to the index
    index.add(embeddings)

    # Perform range search to find all neighbors within a similarity threshold
    result = index.range_search(embeddings, similarity_threshold)

    # Extract result components: lims indicate result ranges per query, D is distances, I are indices
    lims, D, I = result

    # Initialize a set of indices to keep and a set for visited indices
    keep = np.ones(len(embeddings), dtype=bool)
    visited = np.zeros(len(embeddings), dtype=bool)

    # Process the results of the range search to deduplicate embeddings
    for i in range(len(embeddings)):
        if visited[i]:  # If already handled, continue
            continue

        # Get the start and end of the neighbors of the i-th query from lims
        start_idx, end_idx = lims[i], lims[i + 1]
        neighbors = I[start_idx:end_idx]
        distances = D[start_idx:end_idx]

        # Exclude the embedding itself (distance 0 or self-index i)
        neighbors = neighbors[neighbors != i]

        # Mark visited for all neighbors
        visited[neighbors] = True

        # Keep only the current embedding (i-th) and mark rest as duplicates
        keep[neighbors] = False

    # Return only the unique embeddings based on the `keep` array
    unique_indices = np.where(keep)[0]
    return embeddings[unique_indices], unique_indices


def faiss_deduplicate_mr(
    embeddings: np.ndarray,
    max_workers=cpu_count(),
    batch_size=100_000,
    similarity_threshold=0.9,
    index_class=faiss.IndexFlatIP,
    index_args=[],
):
    num_embeddings = embeddings.shape[0]
    batch_starts = list(range(0, num_embeddings, batch_size))

    # Создаем список батчей, которые нужно обработать
    batches = [
        embeddings[start : min(start + batch_size, num_embeddings)]
        for start in batch_starts
    ]

    all_unique_embeddings = []
    all_unique_indices = []

    # Используем ThreadPoolExecutor для параллельного выполнения задач
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_start = {
            executor.submit(
                _faiss_deduplicate_single_v3,
                batch,
                similarity_threshold,
                index_class,
                index_args,
            ): start
            for start, batch in zip(batch_starts, batches)
        }

        # tqdm для отслеживания выполнения параллельных задач
        for future in tqdm(
            as_completed(future_to_start),
            total=len(future_to_start),
            desc="Processing batches",
            unit="batch",
        ):
            batch_start = future_to_start[future]
            unique_embeddings, unique_indices = future.result()

            # Shift the local indices by the starting index of the batch to map back to global indices
            unique_indices_global = unique_indices + batch_start

            all_unique_embeddings.append(unique_embeddings)
            all_unique_indices.append(unique_indices_global)

    # Объединяем результаты
    # all_unique_embeddings = np.vstack(all_unique_embeddings)
    all_unique_indices = np.concatenate(all_unique_indices)

    return embeddings[all_unique_indices], all_unique_indices


def faiss_deduplicate_mr_multistep(
    embeddings: np.ndarray,
    steps_count=3,
    max_workers=cpu_count(),
    batch_size=100_000,
    similarity_threshold=0.9,
    index_class=faiss.IndexFlatIP,
    index_args=[],
):
    progress_indicies_mapping = np.arange(len(embeddings))
    progress_embeddings = embeddings
    sizes_history = [len(embeddings)]

    for step in tqdm(
        range(steps_count), desc="Running global dedup step", total=steps_count
    ):
        shuffled_embeddings, shuffled_indices = shuffle_matrix_with_mapping(
            progress_embeddings
        )
        progress_indicies_mapping = progress_indicies_mapping[shuffled_indices]
        progress_embeddings, unique_indices = faiss_deduplicate_mr(
            shuffled_embeddings.astype(np.float32),
            max_workers=max_workers,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
            index_class=index_class,
            index_args=index_args,
        )
        sizes_history.append(len(progress_embeddings))
        progress_indicies_mapping = progress_indicies_mapping[unique_indices]

    return progress_embeddings, progress_indicies_mapping, sizes_history
