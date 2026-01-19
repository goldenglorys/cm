"""
Clustering Tool for Exercise 2 - EXTENDED VERSION
Clustering Methods Course - University of Eastern Finland

This tool extends Exercise 1 with:
1. Nearest neighbor search
2. Optimal partition
3. Centroid calculation function
4. K-means algorithm
5. Activity tracking for centroids
"""

import numpy as np
import random
import urllib.request


class ClusteringTool:
    def __init__(self):
        self.data = None
        self.centroids = None
        self.partition = None
        self.k = 0
        self.iteration_history = []  # For tracking SSE and activity
        
    def read_data_from_url(self, url):
        """Read data from URL (txt format from cs.uef.fi/sipu/datasets/)"""
        try:
            print(f"Reading data from: {url}")
            response = urllib.request.urlopen(url)
            data_lines = response.read().decode('utf-8').strip().split('\n')
            
            data_list = []
            for line in data_lines:
                if not line.strip():
                    continue
                values = line.strip().split()
                data_list.append([float(v) for v in values])
            
            self.data = np.array(data_list)
            print(f"Data loaded: {self.data.shape[0]} points, {self.data.shape[1]} dimensions")
            return self.data
            
        except Exception as e:
            print(f"Error reading data: {e}")
            return None
    
    def read_data_from_file(self, filepath):
        """Read data from local file (txt format)"""
        try:
            print(f"Reading data from file: {filepath}")
            self.data = np.loadtxt(filepath)
            print(f"Data loaded: {self.data.shape[0]} points, {self.data.shape[1]} dimensions")
            return self.data
            
        except Exception as e:
            print(f"Error reading data: {e}")
            return None
    
    # ========================================================================
    # EXERCISE 1 FUNCTIONS (from previous week)
    # ========================================================================
    
    def dummy_clustering(self, k):
        """
        Dummy clustering algorithm from Exercise 1:
        (a) Select k random data points as centroids
        (b) Assign random partition labels (1 to k)
        """
        if self.data is None:
            print("Error: No data loaded!")
            return
        
        self.k = k
        n_points = self.data.shape[0]
        
        # (a) Select k random data points as centroids
        random_indices = random.sample(range(n_points), k)
        self.centroids = self.data[random_indices].copy()
        
        # (b) Assign random partition labels (1 to k)
        self.partition = np.array([random.randint(1, k) for _ in range(n_points)])
        
        print(f"Dummy clustering completed: {k} clusters")
        print(f"Selected random centroids from indices: {random_indices}")
    
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    # ========================================================================
    # EXERCISE 2 - TASK 1: Nearest Neighbor Search and Optimal Partition
    # ========================================================================
    
    def nearest_neighbor(self, point, search_data):
        """
        Find the nearest neighbor to a point in a dataset
        
        Args:
            point: Data point (numpy array)
            search_data: Dataset to search in (can be data or centroids)
            
        Returns:
            index: Index of nearest neighbor
            distance: Distance to nearest neighbor
        """
        min_distance = float('inf')
        nearest_idx = -1
        
        for i in range(len(search_data)):
            dist = self.euclidean_distance(point, search_data[i])
            if dist < min_distance:
                min_distance = dist
                nearest_idx = i
        
        return nearest_idx, min_distance
    
    def optimal_partition(self, data, centroids):
        """
        Generate optimal partition by assigning each data point 
        to its nearest centroid
        
        Args:
            data: Dataset (numpy array)
            centroids: Set of centroids (numpy array)
            
        Returns:
            partition: Array of cluster labels (1 to k)
        """
        n_points = len(data)
        partition = np.zeros(n_points, dtype=int)
        
        for i in range(n_points):
            nearest_idx, _ = self.nearest_neighbor(data[i], centroids)
            partition[i] = nearest_idx + 1  # Labels are 1 to k
        
        return partition
    
    def calculate_sse(self, data=None, centroids=None, partition=None):
        """
        Calculate Sum of Squared Errors
        
        Args:
            data: Dataset (uses self.data if None)
            centroids: Centroids (uses self.centroids if None)
            partition: Partition (uses self.partition if None)
            
        Returns:
            SSE value
        """
        if data is None:
            data = self.data
        if centroids is None:
            centroids = self.centroids
        if partition is None:
            partition = self.partition
            
        if data is None or centroids is None or partition is None:
            print("Error: Missing data, centroids, or partition!")
            return None
        
        sse = 0
        for i in range(len(data)):
            centroid_idx = partition[i] - 1  # Convert from 1-indexed to 0-indexed
            squared_distance = np.sum((data[i] - centroids[centroid_idx]) ** 2)
            sse += squared_distance
        
        return sse
    
    # ========================================================================
    # EXERCISE 2 - TASK 2: Centroid Function and K-means
    # ========================================================================
    
    def calculate_centroid(self, data_points):
        """
        Calculate centroid (mean) of a set of data points
        
        Args:
            data_points: Array of data points
            
        Returns:
            centroid: Mean vector across all dimensions
        """
        if len(data_points) == 0:
            return None
        
        # Calculate mean for each dimension independently
        n_dims = data_points.shape[1]
        centroid = np.zeros(n_dims)
        
        for d in range(n_dims):
            centroid[d] = np.mean(data_points[:, d])
        
        return centroid
    
    def centroid_step(self, data, partition, k):
        """
        K-means centroid update step: calculate new centroids 
        based on current partition
        
        Args:
            data: Dataset
            partition: Current partition (cluster assignments)
            k: Number of clusters
            
        Returns:
            new_centroids: Updated centroids
        """
        n_dims = data.shape[1]
        new_centroids = np.zeros((k, n_dims))
        
        for cluster_id in range(1, k + 1):
            # Find all points in this cluster
            cluster_points = data[partition == cluster_id]
            
            if len(cluster_points) > 0:
                new_centroids[cluster_id - 1] = self.calculate_centroid(cluster_points)
            else:
                # If cluster is empty, keep the old centroid or reinitialize
                print(f"Warning: Cluster {cluster_id} is empty")
                new_centroids[cluster_id - 1] = new_centroids[cluster_id - 1]
        
        return new_centroids
    
    def kmeans(self, k, max_iterations=100, tolerance=1e-6, verbose=True):
        """
        K-means clustering algorithm
        
        Args:
            k: Number of clusters
            max_iterations: Maximum number of iterations
            tolerance: Convergence threshold for SSE
            verbose: Print iteration details
            
        Returns:
            centroids: Final centroids
            partition: Final partition
            sse: Final SSE value
            iterations: Number of iterations performed
        """
        if self.data is None:
            print("Error: No data loaded!")
            return None, None, None, 0
        
        self.k = k
        n_points = self.data.shape[0]
        
        # Initialize: random selection of k data points as centroids
        random_indices = random.sample(range(n_points), k)
        self.centroids = self.data[random_indices].copy()
        
        if verbose:
            print(f"\nK-means starting with k={k}")
            print(f"Initial centroids from indices: {random_indices}")
        
        prev_sse = float('inf')
        
        for iteration in range(max_iterations):
            # Step 1: Assign points to nearest centroids (optimal partition)
            self.partition = self.optimal_partition(self.data, self.centroids)
            
            # Step 2: Calculate SSE
            current_sse = self.calculate_sse()
            
            if verbose:
                print(f"Iteration {iteration + 1}: SSE = {current_sse:.6f}")
            
            # Check convergence
            if abs(prev_sse - current_sse) < tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
            
            # Step 3: Update centroids
            self.centroids = self.centroid_step(self.data, self.partition, k)
            
            prev_sse = current_sse
        
        final_sse = self.calculate_sse()
        
        if verbose:
            print(f"\nK-means completed!")
            print(f"Final SSE: {final_sse:.6f}")
            print(f"Iterations: {iteration + 1}")
        
        return self.centroids, self.partition, final_sse, iteration + 1
    
    # ========================================================================
    # EXERCISE 2 - TASK 3: Activity Tracking
    # ========================================================================
    
    def kmeans_with_activity_tracking(self, k, max_iterations=100, tolerance=1e-6):
        """
        K-means with centroid activity tracking
        
        Tracks which centroids move (are "active") in each iteration
        
        Returns:
            centroids: Final centroids
            partition: Final partition
            sse: Final SSE
            history: List of dicts with iteration statistics
        """
        if self.data is None:
            print("Error: No data loaded!")
            return None, None, None, []
        
        self.k = k
        n_points = self.data.shape[0]
        
        # Initialize
        random_indices = random.sample(range(n_points), k)
        self.centroids = self.data[random_indices].copy()
        
        print(f"\nK-means with Activity Tracking - k={k}")
        print(f"Initial centroids from indices: {random_indices}")
        print("\n" + "="*70)
        print(f"{'Iter':>5} {'SSE':>15} {'Active':>10} {'Active %':>12}")
        print("="*70)
        
        self.iteration_history = []
        prev_sse = float('inf')
        prev_centroids = self.centroids.copy()
        
        for iteration in range(max_iterations):
            # Assign points to nearest centroids
            self.partition = self.optimal_partition(self.data, self.centroids)
            
            # Calculate SSE
            current_sse = self.calculate_sse()
            
            # Count active centroids (those that moved)
            active_count = 0
            for i in range(k):
                if not np.array_equal(self.centroids[i], prev_centroids[i]):
                    active_count += 1
            
            active_percentage = (active_count / k) * 100
            
            # Store iteration statistics
            iter_stats = {
                'iteration': iteration + 1,
                'sse': current_sse,
                'active_count': active_count,
                'active_percentage': active_percentage
            }
            self.iteration_history.append(iter_stats)
            
            # Print statistics
            print(f"{iteration + 1:5d} {current_sse:15.6f} {active_count:10d} {active_percentage:11.2f}%")
            
            # Check convergence
            if abs(prev_sse - current_sse) < tolerance:
                print("="*70)
                print(f"Converged after {iteration + 1} iterations")
                break
            
            # Update centroids
            prev_centroids = self.centroids.copy()
            self.centroids = self.centroid_step(self.data, self.partition, k)
            
            prev_sse = current_sse
        
        print("="*70)
        final_sse = self.calculate_sse()
        print(f"\nFinal SSE: {final_sse:.6f}")
        
        return self.centroids, self.partition, final_sse, self.iteration_history
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def save_centroids(self, filename="centroid.txt"):
        """Save centroids to file"""
        if self.centroids is None:
            print("Error: No centroids to save!")
            return
        
        with open(filename, 'w') as f:
            for centroid in self.centroids:
                line = ' '.join([str(val) for val in centroid])
                f.write(line + '\n')
        
        print(f"Centroids saved to: {filename}")
    
    def save_partition(self, filename="partition.txt"):
        """Save partition to file"""
        if self.partition is None:
            print("Error: No partition to save!")
            return
        
        with open(filename, 'w') as f:
            for label in self.partition:
                f.write(str(label) + '\n')
        
        print(f"Partition saved to: {filename}")
    
    def save_iteration_history(self, filename="iteration_history.txt"):
        """Save iteration history to file"""
        if not self.iteration_history:
            print("Error: No iteration history to save!")
            return
        
        with open(filename, 'w') as f:
            f.write("Iteration\tSSE\t\tActive_Count\tActive_Percentage\n")
            for stats in self.iteration_history:
                f.write(f"{stats['iteration']}\t{stats['sse']:.6f}\t"
                       f"{stats['active_count']}\t{stats['active_percentage']:.2f}\n")
        
        print(f"Iteration history saved to: {filename}")
    
    def calculate_pairwise_distances(self):
        """Calculate pairwise distances between all data points (from Exercise 1)"""
        if self.data is None:
            print("Error: No data loaded!")
            return None
        
        n_points = self.data.shape[0]
        total_distance = 0
        count = 0
        
        print("Calculating pairwise distances...")
        for i in range(n_points):
            for j in range(i + 1, n_points):
                distance = self.euclidean_distance(self.data[i], self.data[j])
                total_distance += distance
                count += 1
        
        avg_distance = total_distance / count
        print(f"Average pairwise distance: {avg_distance:.6f}")
        print(f"Total pairs calculated: {count}")
        
        return avg_distance


def main():
    """Main function demonstrating Exercise 2 functionality"""
    print("=" * 70)
    print("CLUSTERING TOOL - EXERCISE 2")
    print("=" * 70)
    
    tool = ClusteringTool()
    
    # For testing, use a small dataset
    # You can change this to any dataset from http://cs.uef.fi/sipu/datasets/
    url = "http://cs.uef.fi/sipu/datasets/s1.txt"
    tool.read_data_from_url(url)
    
    # Task 1: Test optimal partition with dummy clustering
    print("\n" + "=" * 70)
    print("TASK 1: Optimal Partition from Exercise 1 Centroids")
    print("=" * 70)
    k = 3
    tool.dummy_clustering(k)
    
    # Generate optimal partition for the random centroids
    optimal_part = tool.optimal_partition(tool.data, tool.centroids)
    tool.partition = optimal_part
    
    # Calculate SSE
    sse_optimal = tool.calculate_sse()
    print(f"\nSSE with optimal partition: {sse_optimal:.6f}")
    
    # Task 2: Run K-means algorithm
    print("\n" + "=" * 70)
    print("TASK 2: K-means Algorithm")
    print("=" * 70)
    centroids, partition, sse, iterations = tool.kmeans(k=3, verbose=True)
    
    # Save results
    tool.save_centroids("ex2_centroid.txt")
    tool.save_partition("ex2_partition.txt")
    
    # Task 3: K-means with activity tracking
    print("\n" + "=" * 70)
    print("TASK 3: K-means with Activity Tracking")
    print("=" * 70)
    centroids, partition, sse, history = tool.kmeans_with_activity_tracking(k=3)
    
    # Save iteration history
    tool.save_iteration_history("ex2_iteration_history.txt")
    
    print("\n" + "=" * 70)
    print("Exercise 2 Tasks Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()