import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

class FlowMatchingSearch:
    """
    A search system that uses flow matching to create smooth transitions 
    between the embedding spaces of captions and songs.
    """
    
    def __init__(self, 
                 embedding_model: SentenceTransformer,
                 collection: Any,
                 device: str = None):
        """
        Initialize the flow matching search system.
        
        Args:
            embedding_model: The SentenceTransformer model for creating embeddings
            collection: The ChromaDB collection containing song embeddings
            device: The device to run computations on ('cpu' or 'cuda')
        """
        self.embedding_model = embedding_model
        self.collection = collection
        
        # Check if collection is empty
        if not self.collection.count():
            raise ValueError("Collection is empty")
            
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.embedding_dim = 384 + 15  # Dimension of the embedding vectors
        # bỏ 15 
        # Flow model
        self.flow_model = nn.Sequential(
            nn.Linear(self.embedding_dim + 1, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, self.embedding_dim)
        ).to(self.device)
        
        # Optimizer with gradient clipping
        self.optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=1e-4)
        
    def train_flow(self, 
                   source_embeddings: np.ndarray, 
                   target_embeddings: np.ndarray,
                   num_steps: int = 100):
        """
        Train the flow model to match distributions between source and target embeddings.
        """
        try:
            print(f"\nDebug train_flow:")
            print(f"Source embeddings shape: {source_embeddings.shape}")
            print(f"Target embeddings shape: {target_embeddings.shape}")
            
            # Convert to tensor and ensure correct dimensions
            source_embeddings = torch.tensor(source_embeddings, dtype=torch.float32).to(self.device)
            target_embeddings = torch.tensor(target_embeddings, dtype=torch.float32).to(self.device)
            
            # Ensure both tensors have the same dimension
            if source_embeddings.shape[1] != target_embeddings.shape[1]:
                # If source has more dimensions, truncate
                if source_embeddings.shape[1] > target_embeddings.shape[1]:
                    source_embeddings = source_embeddings[:, :target_embeddings.shape[1]]
                # If target has more dimensions, pad source
                else:
                    padding = torch.zeros(source_embeddings.shape[0], 
                                        target_embeddings.shape[1] - source_embeddings.shape[1],
                                        device=self.device)
                    source_embeddings = torch.cat([source_embeddings, padding], dim=1)
            # source_embeddings = 399 values per 1 embedding ( 384)
            # target_embeddings = 399 ( 384 caption + 15 emotion) values per 1 embedding

            print(f"After alignment:")
            print(f"Source embeddings shape: {source_embeddings.shape}")
            print(f"Target embeddings shape: {target_embeddings.shape}")
            
            # Limit batch size to prevent OOM
            max_batch_size = 32
            batch_size = min(max_batch_size, len(source_embeddings))
            
            for step in range(num_steps):
                indices = np.random.choice(len(source_embeddings), batch_size, replace=False) # source embedding (100x399) vì chọn 100 truy vấn
                #  chọn bừa 32 embedding trong source_embeddings và target_embeddings
                # ví dụ 399 giá trị -> chọn bừa 32 trị đầu (chỉ số)
                x0 = source_embeddings[indices] 
                # x0 có 32 giá trị
                x1 = target_embeddings[indices]
                # x1 có 32 giá trị
              
                t = torch.rand(batch_size, 1).to(self.device)
                xt = (1 - t) * x0 + t * x1
                v_t = x1 - x0
                
                input_t = torch.cat([xt, t], dim=1)
                v_pred = self.flow_model(input_t)
                
                loss = torch.mean((v_pred - v_t) ** 2)
                
                self.optimizer.zero_grad()
                loss.backward()
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                if (step + 1) % 10 == 0:
                    print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.6f}")
                    
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            print(f"Source embeddings shape: {source_embeddings.shape if 'source_embeddings' in locals() else 'not created'}")
            print(f"Target embeddings shape: {target_embeddings.shape if 'target_embeddings' in locals() else 'not created'}")
            raise
            
    def flow_fn(self, t, y):
        """
        The flow function for ODE integration.
        """
        try:
            print(f"\nDebug flow_fn:")
            print(f"Input t: {t}, type: {type(t)}")
            print(f"Input y shape: {y.shape if hasattr(y, 'shape') else 'no shape'}")
            print(f"Input y type: {type(y)}")
            
            # Ensure y is the right shape and type
            y = np.array(y, dtype=np.float32)
            if len(y.shape) == 1:
                y = y.reshape(1, -1)
            print(f"After reshape y shape: {y.shape}")
            
            # Convert to tensor and move to device
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
            t_tensor = torch.tensor([[t]], dtype=torch.float32).to(self.device)
            print(f"y_tensor shape: {y_tensor.shape}")
            print(f"t_tensor shape: {t_tensor.shape}")
            
            # Ensure input dimensions match
            if y_tensor.shape[1] != self.embedding_dim:
                raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {y_tensor.shape[1]}")
            
            # Predict velocity
            with torch.no_grad():
                input_t = torch.cat([y_tensor, t_tensor], dim=1)
                print(f"input_t shape: {input_t.shape}")
                v_t = self.flow_model(input_t)
                print(f"v_t shape: {v_t.shape}")
            
            # Convert back to numpy and ensure correct shape
            v_np = v_t.cpu().numpy()
            if len(v_np.shape) == 2:
                v_np = v_np.squeeze(0)
            print(f"Final v_np shape: {v_np.shape}")
            
            return v_np
            
        except Exception as e:
            print(f"\nError in flow_fn: {str(e)}")
            print(f"y shape: {y.shape if hasattr(y, 'shape') else 'no shape'}")
            print(f"y type: {type(y)}")
            print(f"y content: {y[:10] if hasattr(y, '__getitem__') else 'cannot print content'}")
            raise
    
    def search(self, 
               query_text: str, 
               n_results: int = 10, 
               flow_steps: int = 50,
               visualize: bool = False) -> Dict:
        """
        Search for similar songs using flow matching.
        """
        try:
            print("\nDebug search:")
            # Get and validate query embedding
            query_embedding = self.embedding_model.encode([query_text])[0]
            print(f"Initial query_embedding shape: {query_embedding.shape}")
            
            if len(query_embedding) != 384:
                raise ValueError(f"Expected embedding dimension 384, got {len(query_embedding)}")
            
            # Convert to tensor and add padding
            query_embedding = torch.tensor(query_embedding, dtype=torch.float32, device=self.device)
            query_embedding = torch.cat([query_embedding, torch.zeros(15, device=self.device)])
            print(f"After padding query_embedding shape: {query_embedding.shape}")
            
            # Get sample embeddings
            print("Querying ChromaDB...")
            sample_results = self.collection.query(
                query_embeddings=[query_embedding.cpu().tolist()],
                n_results=100,
                include=["embeddings"]
            )
            
            if "embeddings" not in sample_results or not sample_results["embeddings"]:
                print("Warning: ChromaDB didn't return embeddings. Fetching a subset of all embeddings instead.")
                all_results = self.collection.get(
                    limit=100,
                    include=["embeddings"]
                )
                if "embeddings" not in all_results or not all_results["embeddings"]:
                    raise ValueError("Failed to retrieve embeddings from ChromaDB")
                sample_embeddings = np.array(all_results["embeddings"])
            else:
                sample_embeddings = np.array(sample_results["embeddings"][0])
            print(f"Sample embeddings shape: {sample_embeddings.shape}")
            
            # Ensure sample embeddings have correct dimension
            if sample_embeddings.shape[1] != self.embedding_dim:
                print(f"Adjusting sample embeddings dimension from {sample_embeddings.shape[1]} to {self.embedding_dim}")
                if sample_embeddings.shape[1] > self.embedding_dim:
                    sample_embeddings = sample_embeddings[:, :self.embedding_dim]
                else:
                    padding = np.zeros((sample_embeddings.shape[0], self.embedding_dim - sample_embeddings.shape[1]))
                    sample_embeddings = np.concatenate([sample_embeddings, padding], axis=1)
            
            # Train flow model
            print("Training flow model...")
            expanded_query = np.repeat(query_embedding.cpu().numpy().reshape(1, -1), len(sample_embeddings), axis=0)
            print(f"Expanded query shape: {expanded_query.shape}")
            self.train_flow(expanded_query, sample_embeddings, num_steps=100)
            
            # Integrate ODE
            print("Integrating ODE...")
            t_span = [0.0, 1.0]
            t_eval = np.linspace(0, 1, flow_steps)
            
            # Ensure initial condition is correct shape
            y0 = query_embedding.cpu().numpy()
            if len(y0.shape) == 1:
                y0 = y0.reshape(1, -1)
            print(f"Initial condition y0 shape: {y0.shape}")
            
            solution = solve_ivp(
                fun=lambda t, y: self.flow_fn(t, y), # t bắt đầu từ 0 -> 1 step là 50-> t0 = 0, t1 = 0.02,..., tn = 1
                t_span=t_span,
                y0=y0.squeeze(),
                # y0 ( embedding của caption) là trạng thái ban đầu: dy/dt = flow model(y,t)
                #  -> sự biến đổi của embeding caption là mạng flow model theo t
                method='RK45',
                t_eval=t_eval
            )

            print(f"Solution shape: {solution.y.shape}")
            
            transformed_query = solution.y[:, -1]
            print(f"Transformed query shape: {transformed_query.shape}")
            
            if visualize:
                print("Visualizing flow...")
                self._visualize_flow(solution, query_embedding, transformed_query)
            
            # Query with transformed embedding
            print("Querying with transformed embedding...")
            results = self.collection.query(
                query_embeddings=[transformed_query.tolist()],
                n_results=n_results
            )
            
            return results
            
        except Exception as e:
            print(f"\nError during search: {str(e)}")
            print(f"Query embedding shape: {query_embedding.shape if hasattr(query_embedding, 'shape') else 'no shape'}")
            print(f"Sample embeddings shape: {sample_embeddings.shape if 'sample_embeddings' in locals() else 'not created'}")
            print(f"Error type: {type(e)}")
            print(f"Error location: {e.__traceback__.tb_lineno}")
            raise
            
    def _visualize_flow(self, solution, query_embedding, transformed_query):
        """
        Visualize the flow trajectory using PCA.
        """
        try:
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=2)
            all_points = solution.y.T
            pca.fit(all_points)
            
            projected_trajectory = pca.transform(all_points)
            projected_query = pca.transform([query_embedding])
            projected_transformed = pca.transform([transformed_query])
            
            plt.figure(figsize=(10, 8))
            plt.plot(projected_trajectory[:, 0], projected_trajectory[:, 1], 'b-', alpha=0.7)
            
            plt.scatter(projected_query[:, 0], projected_query[:, 1], color='green', s=100, label='Original Query')
            plt.scatter(projected_transformed[:, 0], projected_transformed[:, 1], color='red', s=100, label='Transformed Query')
            
            for i in range(0, len(projected_trajectory) - 1, len(projected_trajectory) // 10):
                plt.arrow(
                    projected_trajectory[i, 0],
                    projected_trajectory[i, 1],
                    projected_trajectory[i + 1, 0] - projected_trajectory[i, 0],
                    projected_trajectory[i + 1, 1] - projected_trajectory[i, 1],
                    head_width=0.02,
                    head_length=0.03,
                    fc='blue',
                    ec='blue'
                )
            
            plt.title('Flow Matching Trajectory in Embedding Space (PCA)')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Create output directory if it doesn't exist
            os.makedirs('output', exist_ok=True)
            plt.savefig('output/flow_trajectory.png')
            plt.close()
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            raise

# Usage function for the UI
def flow_matching_search(query_text, embedding_model, collection, n_results=10):
    """
    Perform flow matching search for integration into the Streamlit UI.
    
    Args:
        query_text: The query text (e.g., image caption)
        embedding_model: The SentenceTransformer model
        collection: The ChromaDB collection
        n_results: Number of results to return
        
    Returns:
        Dictionary containing search results
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_system = FlowMatchingSearch(embedding_model, collection, device)
    return search_system.search(query_text, n_results=n_results)

# Test code
if __name__ == "__main__":
    print("Starting test of FlowMatchingSearch...")
    
    try:
        # Initialize embedding model
        print("Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        print("Initializing ChromaDB...")
        import chromadb
        client = chromadb.Client()
        
        # Create or get collection
        print("Creating/getting collection...")
        try:
            collection = client.get_collection("test_collection")
            print("Using existing collection")
        except:
            collection = client.create_collection("test_collection")
            print("Created new collection")
            
            # Add some test data
            print("Adding test data to collection...")
            test_texts = ["happy song", "sad song", "energetic song", "calm song"]
            test_embeddings = model.encode(test_texts).tolist()
            collection.add(
                embeddings=test_embeddings,
                documents=test_texts,
                ids=[f"doc_{i}" for i in range(len(test_texts))]
            )
            print(f"Added {len(test_texts)} test documents")
        
        # Test search
        print("\nTesting search functionality...")
        search_system = FlowMatchingSearch(model, collection)
        results = search_system.search("happy energetic song", n_results=2, visualize=True)
        
        print("\nSearch results:")
        if "documents" in results:
            for i, doc in enumerate(results["documents"][0]):
                print(f"{i+1}. {doc}")
        else:
            print("No documents found in results")
            print("Results keys:", results.keys())
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        import traceback
        traceback.print_exc()