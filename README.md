# Scaled Dot-Product Attention from Scratch

This project implements the core scaled dot-product attention mechanism and a multi-head attention block using only **NumPy**. It demonstrates the inner workings of Transformer attention without relying on deep learning frameworks like PyTorch or TensorFlow.

## Project Structure

* `Scaled_Dot_Product_Attention.ipynb`: The main notebook containing the implementation, visualization, and testing logic.
* `README.md`: Project documentation.

## Features Implemented

1.  **Scaled Dot-Product Attention**:
    * Matrix multiplication of Query ($Q$), Key ($K$), and Value ($V$).
    * Scaling by $\frac{1}{\sqrt{d_k}}$.
    * Numerically stable Softmax normalization.
2.  **Multi-Head Attention**:
    * Splitting inputs into multiple heads.
    * Parallel processing of heads via matrix operations.
    * Concatenation and final linear projection.
3.  **Masking Support**:
    * Support for arbitrary masks (e.g., Causal/Look-ahead masks) to prevent attending to specific tokens.
4.  **Positional Encodings**:
    * Sinusoidal encodings added to input embeddings to retain sequence order.
5.  **Visualization**:
    * Heatmap generation to inspect attention weights.

## Requirements

* Python 3.x
* NumPy
* Matplotlib

## How to Run

1.  Open `Scaled_Dot_Product_Attention.ipynb` in Jupyter Notebook or Google Colab.
2.  Execute the cells sequentially.
3.  **Verify Outputs**:
    * **Shapes**: Ensure tensor shapes remain consistent (e.g., `(Batch, Seq_Len, Dim)`).
    * **Attention Weights**: The unmasked visualization will show global attention.
    * **Masking Test**: The final section demonstrates causal masking, where the attention heatmap should appear lower-triangular (tokens cannot attend to future positions).

## How to Extend This Code

To build a full Transformer model from this base, follow these steps:

1.  **Add Backpropagation**: Currently, this is a forward-pass inference model. You would need to implement the gradients for the linear layers (`WQ`, `WK`, `WV`, `WO`) manually or wrap the NumPy operations in a framework like PyTorch/JAX for autograd.
2.  **Decoder Implementation**: The current block functions as an Encoder layer. To make a Decoder:
    * Add a second attention sub-layer that performs cross-attention (Queries from decoder, Keys/Values from encoder).
    * Enforce the causal mask on the self-attention sub-layer.
3.  **Training Loop**: Implement a training loop with a loss function (e.g., Cross-Entropy) and an optimizer (e.g., Adam) to update the weights.
4.  **Tokenization**: Replace the toy dictionary embeddings with a real tokenizer (BPE or WordPiece) and a larger learned embedding matrix.
