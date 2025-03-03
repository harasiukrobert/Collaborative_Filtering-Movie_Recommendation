## MovieLens Collaborative Filtering Recommendation System

This project implements a collaborative filtering recommendation system using PyTorch Lightning and the MovieLens dataset.  The system predicts user ratings for movies based on user and item embeddings. It explores the impact of incorporating movie genre information on recommendation performance.

**Files:**

*   **`dataset_with_genre.py`**:  Defines the `MovieLensDataset` class for loading and processing the MovieLens data, *including* genre information.  This dataset extracts user ID, item ID, rating, and a one-hot encoded representation of the movie's genres.
*   **`dataset_without_genre.py`**: Defines a similar `MovieLensDataset` class, but *excludes* genre information.  It only extracts user ID, item ID, and rating.
*   **`main_with_genre.py`**:  The main script for training and evaluating the collaborative filtering model *with* genre information.  It utilizes the `Collaborative` model (defined below) and `MovieLensDataset` (from `dataset_with_genre.py`).
*   **`main_without_genre.py`**: The main script for training and evaluating the collaborative filtering model *without* genre information.  It's structurally very similar to `main_with_genre.py`, but uses a modified `Collaborative` model (not shown, but lacking genre-related components) and `MovieLensDataset` (from `dataset_without_genre.py`).

**Model (`Collaborative` in `main_with_genre.py`):**

The core model is a PyTorch Lightning module (`pl.LightningModule`) named `Collaborative`. It uses an embedding-based approach:

1.  **Embeddings:**
    *   `movie_embedding`:  Learns a 256-dimensional embedding for each movie (3952 movies).
    *   `user_embedding`: Learns a 256-dimensional embedding for each user (6040 users).
    *   `genere_embedding`: Learns a 256-dimensional embedding for each genre (19 genres). This is used only in the `main_with_genre.py` version.

2.  **Forward Pass:**
    *   Takes as input a movie ID, user ID, and a genre tensor (one-hot encoded genre vector, only in the with-genre version).
    *   Looks up the corresponding embeddings.
    *   *With Genre*:  Averages the genre embeddings along the sequence dimension (to handle multiple genres per movie).
    *   Concatenates the movie, user, and (optionally) genre embeddings.
    *   Passes the concatenated vector through two fully connected layers (`fc1`, `fc2`) with ReLU activation and dropout (p=0.2) in between.
    *   Outputs a single value representing the predicted rating.

3.  **Training:**
    *   Uses Mean Squared Error Loss (`nn.MSELoss`) to compare predicted ratings with actual ratings.
    *   Calculates Root Mean Squared Error (RMSE) for both training and validation.
    *   Uses the Adam optimizer with a learning rate of 0.001 and weight decay of 1e-5.
    *   Trains the model using 5-fold cross-validation.
    *   Utilizes PyTorch Lightning's `ModelCheckpoint` callback to save the best model based on validation loss.
    *   Uses `torch.set_float32_matmul_precision('high')` to use Tensor Cores.

4.  **Validation and Metrics:**
    *   Calculates precision and recall at k=50 for each user.
    *   A prediction is considered "relevant" if the predicted rating is greater than or equal to a threshold (3.0).
    *   A prediction is considered "correctly predicted" if both prediction and rating is greater than or equal to a threshold (3.0)
    *   Logs average precision, recall, training loss (RMSE), and validation loss (RMSE).

**Results (after 100 epochs):**

*   **Model with Genre:**
    *   Precision: 82%
    *   Recall: 56%
    *   Validation Loss (RMSE): 0.99
    *   Training Loss (RMSE): 0.74

*   **Model without Genre:**
    *   Precision: 76%
    *   Recall: 47%
    *   Validation Loss (RMSE): 1.12
    *   Training Loss (RMSE): 0.72

**Conclusion:**

The results indicate that incorporating genre information significantly improves the performance of the collaborative filtering model.  The model with genre achieves higher precision and recall, as well as a lower validation loss, demonstrating that genre context helps the model make more accurate and relevant recommendations.
